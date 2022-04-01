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
        self.threshold = None

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
            raise TypeError(f"'predictions' must either be a string of the file path or a pandas data frame")

    def set_threshold(self, criteria: str, above_or_below: str, value: float):
        """Threshold on which will be decided if the prediction is a true or false positive."""
        self.threshold = dict()
        implemented_criteria = ["IoU", "distance"]
        accepted_aob = ["above", "below"]
        if criteria not in implemented_criteria:
            raise NotImplementedError(f"Criteria {criteria} is not implemented. Implemented criteria are "
                                      f"{implemented_criteria}.")
        if above_or_below not in accepted_aob:
            raise ValueError(f"'above_or_below' must be one of '[above, below]' but was {accepted_aob}.")
        self.threshold["criteria"] = criteria
        self.threshold["type"] = above_or_below
        self.threshold["value"] = value

    def get_criteria_mean(self, criteria: str or list[str]) -> float or np.ndarray:
        """Criteria might be 'ap' or 'distance'."""
        implemented_early_stopper_criteria = {"ap": self.__get_mean_of_ap,
                                              "distance": self.__get_mean_core_distance}
        assert criteria in implemented_early_stopper_criteria.keys(), \
            f"Not implemented early stopper criteria {criteria}. Implemented are" \
            f" {list(implemented_early_stopper_criteria.keys())}."
        criteria_mean = implemented_early_stopper_criteria[criteria]
        return criteria_mean()

    def get_all_criteria_mean(self) -> tuple[float or np.ndarray, float or np.ndarray]:
        """Returns mean of ap and mean of core distance"""
        return self.__get_mean_of_ap(), self.__get_mean_core_distance()

    def get_labels(self) -> dict:
        criteria_function_available = {"IoU": self.get_IoU}
        criteria_function = criteria_function_available[self.threshold["criteria"]]
        predictions_labeled = dict()
        for img_idx in self.df_predictions["img_idx"].unique():
            predictions_labeled[img_idx] = list()
            df_tmp_predictions = self.df_predictions.loc[self.df_predictions["img_idx"] == img_idx]
            bboxes_truth, _ = helper.pd_row2(self.df_truth, img_idx, "list")
            info_prediction = helper.df_predictions_to_list(df_tmp_predictions)
            assignment = self.__get_assignments(bboxes_truth, info_prediction, criteria_function)
            img_labels, _ = self.__assign_label(assignment)
            for label, bbox_pairs in zip(img_labels, assignment.values()):
                for bbox_pair in bbox_pairs:
                    prediction_number = bbox_pair[1]
                    predictions_labeled[img_idx].append((label, info_prediction[prediction_number][1]))
        return predictions_labeled

    def __get_mean_of_ap(self) -> np.ndarray:
        assert type(self.threshold) is not None, "'threshold' is not initialised. Use 'set_threshold' before using " \
                                                 "'get_mean_of_ap'."
        criteria_function_available = {"IoU": self.get_IoU}
        criteria_function = criteria_function_available[self.threshold["criteria"]]
        average_precision = list()
        for img_idx in self.df_predictions["img_idx"].unique():
            df_tmp_predictions = self.df_predictions.loc[self.df_predictions["img_idx"] == img_idx]
            bboxes_truth, _ = helper.pd_row2(self.df_truth, img_idx, "list")
            info_prediction = helper.df_predictions_to_list(df_tmp_predictions)
            assignment = self.__get_assignments(bboxes_truth, info_prediction, criteria_function)
            labels, scores = self.__assign_label(assignment)
            if np.count_nonzero(labels) > 1:
                average_precision.append(average_precision_score(labels, scores))
            else:
                average_precision.append(0)
        return np.mean(average_precision)

    def __get_mean_core_distance(self):
        assert type(self.threshold) is not None, "'threshold' is not initialised. Use 'set_threshold' before using " \
                                                 "'get_mean_core_distance'."
        criteria_function_available = {"IoU": self.get_IoU}
        criteria_function = criteria_function_available[self.threshold["criteria"]]
        mean_core_distances, n_imgs_no_detection = list(), 0
        for img_idx in self.df_predictions["img_idx"].unique():
            df_tmp_predictions = self.df_predictions.loc[self.df_predictions["img_idx"] == img_idx]
            bboxes_truth, _ = helper.pd_row2(self.df_truth, img_idx, "list")
            info_prediction = helper.df_predictions_to_list(df_tmp_predictions)
            assignment = self.__get_assignments(bboxes_truth, info_prediction, criteria_function)
            mean_core_distance = self.__get_mean_distances_of_true_positives(assignment, bboxes_truth, img_idx)
            if mean_core_distance is not None:
                mean_core_distances.append(mean_core_distance)
            else:
                n_imgs_no_detection += 1
        return np.mean(mean_core_distances), n_imgs_no_detection

    def __get_assignments(self,
                          bboxes_truth: list[list],
                          info_prediction: list[tuple[float, list]],
                          criteria_function) -> dict:
        """Per image. Returns: {criteria_value: ([truth_bbox_id, pred_bbox_id], [], ...), ...)
        bboxes_truth: [[bbox1], [bbox2], ...]
        info_prediction: [(score_bbox1, [bbox1]), (score_bbox2, [bbox2]), ...]
        criteria_function: function that takes one truth bboxes and one prediction bbox as input."""
        needed_keys, fixed_assignments = ["type", "value"], list()
        if not all(key in list(self.threshold.keys()) for key in needed_keys):
            raise ValueError(f"Parameter 'self.threshold' needs to be a dict with keys {needed_keys} but were "
                             f" {list(self.threshold.keys())}.")
        extreme = min if self.threshold["type"] == "below" else max
        self.threshold_value = self.threshold["value"] if self.threshold["value"] is not None else extreme([0, 10e10])
        prediction_numbered = {tuple(bbox_information[1]): i for i, bbox_information in enumerate(info_prediction)}
        df_assignment = pd.DataFrame(columns=["value", "bbox_pair"])

        iter_prediction = prediction_numbered
        for id_truth_bbox, bbox_truth in enumerate(bboxes_truth):
            for bbox_prediction, id_prediction_bbox in copy(iter_prediction).items():
                value = criteria_function(bbox_prediction, bbox_truth)
                df_assignment = df_assignment.append({"value": value, "bbox_pair": (id_truth_bbox, id_prediction_bbox)},
                                                     ignore_index=True)

                case_1 = self.threshold["type"] == "above" and value >= self.threshold_value
                case_2 = self.threshold["type"] == "below" and value <= self.threshold_value
                if case_1 or case_2:  # prediction has fixed assigned truth bbox. No need to check that pred bbox in
                    # the future
                    iter_prediction.pop(bbox_prediction)
        return self.__clean_threshold_assignments(df_assignment, extreme)

    def __assign_label(self,
                       assignment: dict) -> tuple[list, list]:
        """If the criteria value is above/below a certain threshold, the prediction will be labeled as true positive,
        else as false positive."""
        labels, scores = list(), list()
        threshold_type, threshold_value = self.threshold["type"], self.threshold["value"]
        for criteria_value in assignment.keys():
            case_1 = threshold_type == "above" and criteria_value >= threshold_value
            case_2 = threshold_type == "below" and criteria_value <= threshold_value
            labels.append(1 if case_1 or case_2 else 0)
            scores.append(criteria_value)
        return labels, scores

    def __get_mean_distances_of_true_positives(self, assignment: dict, bboxes_truth: list[list[float]],
                                               img_idx: int) -> np.ndarray or None:
        bboxes_truth_numbered = {i: bbox for i, bbox in enumerate(bboxes_truth)}
        distances = list()
        df_predictions = self.df_predictions.loc[self.df_predictions["img_idx"] == img_idx]
        df_predictions = df_predictions.reset_index(drop=True)
        for threshold_value, bbox_pairs in assignment.items():
            if threshold_value >= self.threshold["value"]:
                for bbox_pair in bbox_pairs:
                    bbox_truth = bboxes_truth_numbered[bbox_pair[0]]
                    bbox_prediction = df_predictions.loc[bbox_pair[1]]["bbox"]
                    distances.append(self.get_core_distance(bbox_truth, bbox_prediction))
        distances = np.mean(distances) if len(distances) > 0 else None
        return distances

    @staticmethod
    def __clean_threshold_assignments(df_assignment: pd.DataFrame, extreme_function) -> dict:
        """
        Returns a dict in which the keys are the criteria value and the values are a list of tuples of the
        combination of the prediction id and the truth id.
        :param assignment_dict:
        :param fixed_couple: (id_truth_bbox, id_prediction_bbox)
        :return:
        """
        break_loop_at, counter = 10e3, 0
        fixed_assignments = dict()
        while df_assignment.shape[0] > 0:
            best_value = extreme_function(df_assignment["value"].unique().tolist())
            fixed_assignments[best_value] = list()
            best_row_ids = df_assignment.loc[df_assignment["value"] == best_value].index.tolist()
            rows_to_drop, used_prediction_id = list(), list()
            for best_row_idx in best_row_ids:
                fixed_assignment = df_assignment.loc[best_row_idx]["bbox_pair"]
                if fixed_assignment[1] in used_prediction_id:
                    continue
                else:
                    used_prediction_id.append(fixed_assignment[1])
                fixed_assignments[best_value].append(fixed_assignment)
                for row_idx, pair in enumerate(df_assignment["bbox_pair"].tolist()):
                    if pair[1] == fixed_assignment[1]:
                        rows_to_drop.append(row_idx)
            df_assignment = df_assignment.drop(rows_to_drop)
            df_assignment = df_assignment.reset_index(drop=True)
            counter += 1
            if counter == break_loop_at:
                raise ValueError(f"Infinite loop warning. If there are indeed more than break_loop={break_loop_at} "
                                 f"unique truth-prediction assignments then raise the 'break_loop' value.")
        return fixed_assignments
