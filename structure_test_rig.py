from vortex_detection import data_prep as prep
from helper_functions import Helper
import pandas as pd
import torch
import os
from structure_train_rig import UtilityModel as util_model, CustomDataset
import ast
helper = Helper()


class TestRig(util_model):
    def __init__(self,
                 orders: prep.Orders,
                 root_dir: str):
        util_model.__init__(self)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.orders = orders
        self.dir_root = root_dir

    def all(self,
        num_classes: int = 2,
        transform=None):
        transform = transform if transform is not None else self.get_transform(False)
        n_test_orders = self.orders.get_number_of_orders("test")
        for idx_test, dir_ids in enumerate(self.orders.get_all_parents_dir_idx("test")):
            helper.print_progress(idx_test, n_test_orders, "Testing")
            self.__test_dir_ids(dir_ids, transform, num_classes)

    def test_after_train(self,
                         dir_ids: list[dict],
                         transform,
                         num_classes: int=2):
        for dir_indices in dir_ids:
            self.__test_dir_ids(dir_indices, transform, num_classes)

    def test_validation(self,
                        model,
                        data_set: CustomDataset):
        predictions = self.__predictions_to_df_dict(self.__test_dataset(model, data_set))
        return pd.DataFrame(predictions)

    def __test_dir_ids(self,
                       dir_ids: dict,
                       transform,
                       num_classes: int):
        current_data_dir = self.dir_root + f"/data/{dir_ids['data']}"
        dir_data_set = current_data_dir + "/imgs"
        dir_train = current_data_dir + f"/{dir_ids['train']}"
        dir_test = dir_train + f"/{dir_ids['test']}"

        if self.orders.get_value_for(dir_ids, "epochs") == None:
            print(f"Test directory {dir_ids['test']} does not have a trained model.")
            return

        if os.path.isdir(dir_test):
            if os.path.isdir(dir_test):
                print(f"Already tested {dir_ids['test']}")
                return
            else:
                print(f"Unfinished training in train_dir_idx: {dir_ids['test']}. Repeating that order.")
        else:
            helper.create_dir(dir_test)

        file_bbox = current_data_dir + "/bbox.dat"
        prediction_file = dir_test + "/predictions.dat"

        n_train = self.orders.get_value_for(dir_ids, "nImgTrain")
        val_info = ast.literal_eval(self.orders.get_value_for(dir_ids, "validationInfo"))
        n_val = val_info[0]*val_info[1]
        n_train_val = n_train+n_val
        n_test = n_train_val+self.orders.get_value_for(dir_ids, "nImgTest")
        data_set = CustomDataset(dir_data_set, file_bbox, (n_train_val, n_test), transform)

        model = self.get_model(num_classes)
        model.to(self.device)
        model.load_state_dict(torch.load(dir_train + "/model.pt"))

        predictions = self.__predictions_to_df_dict(self.__test_dataset(model, data_set))
        pd.DataFrame(predictions, index=None).to_csv(prediction_file, index=False)

    def __test_dataset(self,
                       model,
                       data_set: CustomDataset) -> dict:
        predictions = dict()
        model.to(self.device)
        model.eval()
        for data_set_object in data_set:
            img_idx, prediction = self.__test_model(model, data_set_object)
            predictions[img_idx] = prediction
        return predictions

    @torch.no_grad()
    def __test_model(self,
                     model,
                     data_set_object) -> tuple[int, dict]:
        translate_prediction_keys = {"boxes": "bbox", "labels": "label", "scores": "score"}
        img, img_info = data_set_object
        img = img.to(self.device)
        with torch.no_grad():
            iter_prediction = model([img])
        prediction = dict()
        if self.device.type == "cuda":
            img_idx = img_info["image_id"].tolist()
            for key, value in iter_prediction[0].items():
                prediction[translate_prediction_keys[key]] = value.tolist()
        else:
            img_idx = img_info["image_id"]
        return img_idx[0], prediction

    @staticmethod
    def __predictions_to_df_dict(predictions) -> dict:
        first_key = [*predictions][0]
        prediction_criteria = list(predictions[first_key].keys())
        predictions_for_df = {column: list() for column in ["img_ids"]+prediction_criteria}
        for img_idx in predictions.keys():
            for prediction_criteria, values in predictions[img_idx].items():
                predictions_for_df[prediction_criteria] += values
            for _ in range(len(values)):
                predictions_for_df["img_ids"].append(img_idx)
        return predictions_for_df

