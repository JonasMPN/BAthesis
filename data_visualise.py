from typing import List

import data_prep as prep
from helper_functions import Helper
from PIL import Image, ImageDraw
from data_evaluation import PredictionEvaluator
import pandas as pd

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

    def __vis_dir_ids(self,
                      dir_ids: dict,
                      visualise_as: list[str]):
        current_data_dir = self.dir_root + f"/data/{dir_ids['data']}"
        dir_train = current_data_dir + f"/{dir_ids['train']}"
        dir_test = dir_train + f"/{dir_ids['test']}"
        prediction_file = dir_test + "/predictions.dat"
        file_bbox = current_data_dir + "/bbox.dat"

        self.dir_save = dir_test + "/predictions_visualised"
        helper.create_dir(self.dir_save, overwrite=True)
        self.set_assignment(criteria=self.orders.get_value_for(dir_ids, "assignCriteria"),
                            better=self.orders.get_value_for(dir_ids, "assignBetter"),
                            threshold=self.orders.get_value_for(dir_ids, "assignTpThreshold"))
        self.set_prediction_criteria(criteria=self.orders.get_value_for(dir_ids, "predictionCriteria"),
                                     better=self.orders.get_value_for(dir_ids, "predictionBetter"),
                                     threshold=self.orders.get_value_for(dir_ids, "predictionThreshold"))
        self.set_predictions(prediction_file)
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

