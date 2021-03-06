import os.path
from typing import List

import data_prep as prep
from helper_functions import Helper
from PIL import Image, ImageDraw
from data_evaluation import PredictionEvaluator
import pandas as pd
import ast

helper = Helper()

class Visualise(PredictionEvaluator):
    def __init__(self,
                 orders: prep.Orders,
                 root_dir: str):
        PredictionEvaluator.__init__(self)
        self.orders = orders
        self.dir_root = root_dir
        self.dir_save = None

    def all(self, visualise_as: list[str]=["bbox"]):
        n_test_orders = self.orders.get_number_of_orders("test")
        for idx_test, dir_ids in enumerate(self.orders.get_all_parents_dir_idx("test")):
            helper.print_progress(idx_test, n_test_orders, "Visualising")
            self.__vis_dir_ids(dir_ids, visualise_as)

    def truth(self, working_dir: str=None):
        """
        Plot truth bboxes for a single parameter combination using 'working_dir'. When 'working_dir' is None,
        the bboxes for all orders will be plotted.
        :param working_dir:
        :return:
        """
        if working_dir is not None:
            self.__single_truth(working_dir)
        else:
            for idx_test, dir_ids in enumerate(self.orders.get_all_parents_dir_idx("test")):
                self.__single_truth(dir_ids["data"])

    def __single_truth(self, working_dir: str=None):
        dir_bbox_plot = working_dir + "/imgs_bbox"
        helper.create_dir(dir_bbox_plot)
        file_bbox = working_dir + "/bbox.dat"
        df_bbox = pd.read_csv(file_bbox, index_col=None)
        for img_idx in range(len(os.listdir(working_dir + "/imgs"))):
            bboxes, _ = helper.pd_row2(df_bbox, img_idx, "list")
            img = Image.open(working_dir + f"/imgs/{img_idx}.png").convert("RGB")
            draw = ImageDraw.Draw(img)
            for bbox in bboxes:
                self.__visualise_bbox(draw, bbox, "green")
            img.save(dir_bbox_plot + f"/{img_idx}.png")

    def __vis_dir_ids(self,
                      dir_ids: dict,
                      visualise_as: list[str]):
        current_data_dir = self.dir_root + f"/data/{dir_ids['data']}"
        dir_train = current_data_dir + f"/{dir_ids['train']}"
        dir_test = dir_train + f"/{dir_ids['test']}"
        file_prediction = dir_test + "/predictions.dat"
        file_bbox = current_data_dir + "/bbox.dat"
        if not os.path.isfile(file_prediction):
            return

        self.dir_save = dir_test + "/predictions_visualised"
        helper.create_dir(self.dir_save, overwrite=True)
        assign_criteria = self.orders.get_value_for(dir_ids, "assignCriteria")
        assign_threshold = self.orders.get_value_for(dir_ids, "assignTpThreshold")
        need_size = True if assign_criteria == "distance" else False
        if need_size:
            assign_threshold *= self.__get_img_size(dir_ids)[0]  # distance is now absolut and not normalised
        self.set_assignment(criteria=assign_criteria,
                            better=self.orders.get_value_for(dir_ids, "assignBetter"),
                            threshold=assign_threshold)
        self.set_prediction_criteria(criteria=self.orders.get_value_for(dir_ids, "predictionCriteria"),
                                     better=self.orders.get_value_for(dir_ids, "predictionBetter"),
                                     threshold=self.orders.get_value_for(dir_ids, "predictionThreshold"))
        self.set_predictions(file_prediction)
        self.set_truth(file_bbox)
        predictions_labeled = self.get(["labels"])[0]["labels"]
        self.__visualise_predictions(current_data_dir, pd.read_csv(file_bbox, index_col=None), predictions_labeled,
                                     visualise_as)

    def __visualise_predictions(self,
                                data_dir: str,
                                df_ground_truth: pd.DataFrame,
                                predictions_labeled: dict,
                                visualise_as: list[str]):
        visualiser = {"core": self.__visualise_core, "bbox": self.__visualise_bbox}
        for img_idx, prediction_labeled in predictions_labeled.items():
            img = Image.open(data_dir+f"/imgs/{img_idx}.png").convert("RGB")
            draw = ImageDraw.Draw(img)
            truth_bboxes, _ = helper.pd_row2(df_ground_truth, img_idx, "list")

            labels, predictions = list(), list()
            for pred in prediction_labeled:
                labels.append(pred[0])
                predictions.append(pred[1])

            visualise = [
                {"colour": {1: "green"}, "bboxes": truth_bboxes, "scores":[1 for _ in range(len(truth_bboxes))]},
                {"colour": {0: "red", 1: "orange"}, "bboxes": predictions, "scores": labels}
            ]
            for vis_item in visualise:
                for visualising_type in visualise_as:
                    for bbox, score in zip(vis_item["bboxes"], vis_item["scores"]):
                        visualiser[visualising_type](draw, bbox, vis_item["colour"][score], score)

            img.save(self.dir_save + f"/{img_idx}.png")

    def __visualise_core(self,
                         draw: ImageDraw.Draw,
                         bbox: List[float],
                         colour: str,
                         score: float=None):
        """Ground truth are bounding boxes."""
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        draw.point((x, y), fill=colour)
        if score is not None:
            draw.text((x, y), text=str(score))

    def __visualise_bbox(self,
                         draw: ImageDraw.Draw,
                         bbox: List[float],
                         colour: str,
                         score: float=None,
                         width: int=3):
        draw.rectangle(bbox, outline=colour, width=width)
        if score is not None:
            draw.text((bbox[0], bbox[1]), text=str(score))

    def __get_img_size(self, dir_ids: dict) -> tuple[int, int] or None:
        plotType = self.orders.get_value_for(dir_ids, "plotType")
        if plotType == "vec":
            size = ast.literal_eval(self.orders.get_value_for(dir_ids, "imgSize"))
            width, height = size[0], size[1]
        elif plotType == "col":
            size = ast.literal_eval(self.orders.get_value_for(dir_ids, "nInfoPerAxis"))
            width, height = size[0], size[1]
        else:
            return None
        return width, height