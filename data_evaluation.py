import ast

import pandas as pd
import numpy as np
from typing import List
import helper_functions as hf
from copy import copy
from sklearn.metrics import average_precision_score

helper = hf.Helper()


class BoundingBoxEvaluator:
    def __init__(self):
        pass

    def get_IoU(self, bbox_1: list[float], bbox_2: list[float]) -> float:
        intersection = self.__get_intersection(bbox_1, bbox_2)
        area = self.__get_area(bbox_1) + self.__get_area(bbox_2)
        return intersection / (area - intersection)

    def get_core_distance(self, bbox_1: list[float], bbox_2: list[float]) -> float:
        center_pred = self.__get_center(bbox_1)
        center_truth = self.__get_center(bbox_2)
        return np.sqrt((center_pred[0] - center_truth[0]) ** 2 + (center_pred[1] - center_truth[1]) ** 2)

    @staticmethod
    def __get_intersection(bbox1, bbox2) -> float:
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        return max(0, x_max - x_min) * max(0, y_max - y_min)

    @staticmethod
    def __get_area(bbox: List[float]) -> float:
        dx = bbox[2] - bbox[0]
        dy = bbox[3] - bbox[1]
        return dx * dy

    @staticmethod
    def __get_center(bbox) -> tuple[float, float]:
        bbox = ast.literal_eval(bbox) if type(bbox) == str else bbox
        return ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)


class PredictionEvaluator(BoundingBoxEvaluator):
    def __init__(self):
        BoundingBoxEvaluator.__init__(self)
        self.assignment = None
        self.prediction = None
        self.prediction_score_threshold = None
        self.df_truth = None
        self.df_predictions = None
        self.df_filtered_predictions = None
        self.implemented_criteria = ["ap", "distance", "labels"]

    def set_predictions(self, predictions: str or pd.DataFrame):
        if type(predictions) == str:
            self.df_predictions = pd.read_csv(predictions, index_col=None)
        elif type(predictions) == pd.DataFrame:
            self.df_predictions = predictions
        else:
            raise TypeError(f"'predictions' must either be a string of the file path or a pandas data frame")

    def set_truth(self, truth: str or pd.DataFrame):
        if type(truth) == str:
            self.df_truth = pd.read_csv(truth, index_col=None)
        elif type(truth) == pd.DataFrame:
            self.df_truth = truth
        else:
            raise TypeError(f"'truth' must either be a string of the file path or a pandas data frame")

    def set_assignment(self, criteria: str, better: str, threshold: float):
        """Defines the way the prediction bboxes are assigned to the truth bboxes. Each prediction is assigned to
        the truth box with which it has the best ('better') 'criteria' value. 'threshold' is used to decide whether
        it is a true positive or a false positive."""
        implemented_criteria_function = {"IoU": self.get_IoU, "distance": self.get_core_distance}
        accepted_better = ["above", "below"]
        if criteria not in implemented_criteria_function:
            raise NotImplementedError(f"Criteria {criteria} is not implemented. Implemented criteria are "
                                      f"{implemented_criteria_function.keys()}.")
        if better not in accepted_better:
            raise ValueError(f"'above_or_below' must be one of '[above, below]' but was {better}.")

        self.assignment = {
            "criteria_function": implemented_criteria_function[criteria],
            "better": max if better == "above" else min,
            "threshold": threshold
        }

    def set_prediction_criteria(self, criteria: str, better: str, threshold: float):
        """
        Defines which predictions are used in the test and validation phase. Because only these predictions are used
        in a real-world environment they are also used for the criteria 'distance' and 'labels'.
        :param criteria: Must be a parameter that is associated with each prediction on its own. Therefore,
        it must be a column of the dataframe that is set with 'set_predictions'.
        :param better:
        :param threshold:
        :return:
        """
        possible_criteria_param = ["score"] # todo hardcoded. Should be the columns from the prediction dataframe
        accepted_better = ["above", "below"]
        if criteria not in possible_criteria_param:
            raise NotImplementedError(f"Criteria {criteria} is not implemented. Implemented criteria are "
                                      f"{possible_criteria_param}.")
        if better not in accepted_better:
            raise ValueError(f"'above_or_below' must be one of '[above, below]' but was {better}.")
        self.prediction = {
            "criteria_param": criteria,
            "better": max if better == "above" else min,
            "threshold": threshold
        }

    def get(self, criteria: list[str]) -> tuple[dict, int]:
        """
        Criteria might be 'ap', 'distance' or 'labels'.
        'ap' will be calculated over all predictions.
        'distance' and 'labels' will be calculated only for predictions which scores are above the threshold.
        :param criteria:
        :return:
        """
        assert self.assignment is not None, "Use 'set_assignment' before any type of evaluation."
        assert self.prediction is not None, "Use 'set_prediction_criteria' before any type of evaluation."
        for crit in criteria:
            if crit not in self.implemented_criteria:
                raise ValueError(f"Not implemented criteria {crit}. Implemented are {self.implemented_criteria}.")
        results = {
            "ap": list(),
            "distance": list(),
            "labels": dict()
        }
        n_detected = 0
        for img_idx in self.df_predictions["img_ids"].unique():
            df_tmp_predictions = self.df_predictions.loc[self.df_predictions["img_ids"] == img_idx]
            bboxes_truth, _ = helper.pd_row2(self.df_truth, img_idx, "list")
            info_prediction = helper.df_predictions_to_list(df_tmp_predictions, self.prediction["criteria_param"])
            assignment, prediction_filtered_assignment, n_new_detected = self.__get_assignments(bboxes_truth,
                                                                                           info_prediction)
            n_detected += n_new_detected
            if "ap" in criteria:
                labels, scores = self.__assign_label(assignment)
                if np.count_nonzero(labels) >= 1:
                    results["ap"].append(average_precision_score(labels, scores))
                else:
                    results["ap"].append(0)
            if len(prediction_filtered_assignment) != 0: # else there are no predictions that are better than the score threshold
                if "distance" in criteria:
                    mean_core_distance = self.__get_mean_distance(prediction_filtered_assignment)
                    if mean_core_distance is not None:
                        results["distance"].append(mean_core_distance)
                if "labels" in criteria:
                    results["labels"][img_idx] = list()
                    img_labels, _ = self.__assign_label(prediction_filtered_assignment)
                    for label, bbox_pairs in zip(img_labels, prediction_filtered_assignment.values()):
                        for bbox_pair in bbox_pairs:
                            results["labels"][img_idx].append([label, bbox_pair[1]])
        for crit in ["ap", "distance"]:
            results[crit] = np.mean(results[crit]) if len(results[crit]) != 0 else None
        return results, n_detected

    def get_all_criteria(self) -> tuple[dict, int]:
        """Returns mean of ap and mean of core distance"""
        return self.get(self.implemented_criteria)

    def __get_assignments(self,
                          bboxes_truth: list[list],
                          info_prediction: list[tuple[float, list]]) -> tuple[dict, dict, int]:
        """Per image. Returns: {criteria_value: ([truth_bbox_id, pred_bbox_id], [], ...), ...)
        bboxes_truth: [[bbox1], [bbox2], ...]
        info_prediction: [(prediction_criteria_value1, [bbox1]), (prediction_criteria_value1, [bbox2]), ...]
        criteria_function: function that takes one truth bboxes and one prediction bbox as input."""
        assignment_better, assignment_tp_threshold = self.assignment["better"], self.assignment["threshold"]
        assign_criteria_function = self.assignment["criteria_function"]
        prediction_better, prediction_threshold = self.prediction["better"], self.prediction["threshold"]
        prediction_numbered = {tuple(bbox_information[1]): bbox_information[0] for bbox_information in info_prediction}
        df_assignment = pd.DataFrame(columns=["value", "bbox_pair", "prediction_score"])
        df_filtered_predictions = pd.DataFrame(columns=["value", "bbox_pair", "prediction_score"])
        for bbox_truth in bboxes_truth:
            for bbox_prediction, prediction_score in copy(prediction_numbered).items():
                value = assign_criteria_function(bbox_prediction, bbox_truth)
                to_append = {"value": value, "bbox_pair": (bbox_truth, bbox_prediction), "score": prediction_score}
                if prediction_better(prediction_threshold, prediction_score) == prediction_score:
                    df_filtered_predictions = df_filtered_predictions.append(to_append, ignore_index=True)
                df_assignment = df_assignment.append(to_append, ignore_index=True)
                if assignment_better(assignment_tp_threshold, value) == value:  # prediction has fixed assigned truth
                    #  bbox. No need to check that pred bbox in the future
                    prediction_numbered.pop(bbox_prediction)
        return self.__clean_threshold_assignments(df_assignment)

    def __assign_label(self,
                       assignment: dict) -> tuple[list, list]:
        """If the criteria value is above/below a certain threshold, the prediction will be labeled as true positive,
        else as false positive."""
        labels, scores = list(), list()
        assign_better = self.assignment["better"]
        assign_tp_threshold = self.assignment["threshold"]
        for assign_criteria_value, bbox_information in assignment.items():
            for bbox_pair in bbox_information:
                prediction_score = bbox_pair[1]
                labels.append(1 if assign_better(assign_criteria_value, assign_tp_threshold) == assign_criteria_value else 0)
                scores.append(prediction_score)
        return labels, scores

    def __get_mean_distance(self, assignment: dict) -> np.ndarray or None:
        distances = list()
        if len(assignment) == 0:
            return None
        for assign_criteria_value, bbox_information in assignment.items():
            for bbox_pair in bbox_information:
                distances.append(self.get_core_distance(bbox_pair[0], bbox_pair[1]))
        mean_distance = np.mean(distances)
        return mean_distance

    def __clean_threshold_assignments(self, df_assignment: pd.DataFrame) -> tuple[dict, dict, int]:
        """
        Returns a dict in which the keys are the criteria value and the values are a list of tuples of the
        combination of the prediction id and the truth id.
        :param df_assignment:
        :param best_function:
        :return:
        """
        assign_better, assign_tp_threshold = self.assignment["better"], self.assignment["threshold"]
        prediction_better, prediction_threshold = self.prediction["better"], self.prediction["threshold"]
        break_loop_at, counter = 10e3, 0
        fixed_assignments = dict()
        fixed_filtered_assignments = dict()
        tp_bbox = list()
        while df_assignment.shape[0] > 0:
            best_value = assign_better(df_assignment["value"].unique().tolist())
            fixed_assignments[best_value] = list()
            fixed_filtered_assignments[best_value] = list()
            best_row_ids = df_assignment.loc[df_assignment["value"] == best_value].index.tolist()
            rows_to_drop, used_prediction_bbox = list(), list()
            for row_idx in best_row_ids:
                row = df_assignment.loc[row_idx]
                fixed_assignment, prediction_score= row["bbox_pair"], row["score"]
                truth_bbox, pred_bbox = fixed_assignment[0], fixed_assignment[1]
                if pred_bbox in used_prediction_bbox:
                    continue
                else:
                    used_prediction_bbox.append(pred_bbox)

                if prediction_better(prediction_score, prediction_threshold) == prediction_score:
                    fixed_filtered_assignments[best_value].append([truth_bbox, list(pred_bbox)])
                    if assign_better(best_value, assign_tp_threshold) == best_value:
                        if truth_bbox not in tp_bbox:
                            tp_bbox.append(truth_bbox)

                fixed_assignments[best_value].append([[truth_bbox, list(pred_bbox)], prediction_score])
                for row_idx, pair in enumerate(df_assignment["bbox_pair"].tolist()):
                    if pair[1] == pred_bbox:
                        rows_to_drop.append(row_idx)
            df_assignment = df_assignment.drop(rows_to_drop)
            df_assignment = df_assignment.reset_index(drop=True)
            if len(fixed_filtered_assignments[best_value]) == 0:
                fixed_filtered_assignments.pop(best_value)
            counter += 1
            if counter == break_loop_at:
                raise ValueError(f"Infinite loop warning. If there are indeed more than break_loop={break_loop_at} "
                                 f"unique truth-prediction assignments then raise the 'break_loop' value.")
        return fixed_assignments, fixed_filtered_assignments, len(tp_bbox)
