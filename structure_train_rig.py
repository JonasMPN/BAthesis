import numpy as np
from torch.utils.data import Dataset
import transforms as transf
import torch
import torchvision
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import pandas as pd
from vortex_detection import structure_test_rig as test, data_prep as prep
from vortex_detection.data_evaluation import PredictionEvaluator
from helper_functions import Helper
import ast
from torch.utils.data import DataLoader
from utils import collate_fn
from engine import train_one_epoch
import itertools

helper = Helper()

class CustomDataset(Dataset):
    def __init__(self, root, file_bbox, img_ids: tuple=None, transforms=None):
        self.root = root
        self.df_bbox = pd.read_csv(file_bbox, index_col=None)
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(self.root)))
        self.iter_id = 0
        if img_ids is None:
            self.img_ids = (0, len(self.imgs))
        else:
            self.img_ids = img_ids
            self.imgs = self.imgs[int(img_ids[0]):int(img_ids[1])]

    def __getitem__(self, iterator_idx):
        self.iter_id += 1
        img_path = self.root + f"/{iterator_idx}.png"
        img = Image.open(img_path).convert("RGB")
        boxes, num_objs = helper.pd_row2(self.df_bbox, iterator_idx, "torch")
        labels = torch.ones((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([iterator_idx]),
            "area": area,
            "iscrowd": iscrowd}

        if self.transforms is not None and self.df_bbox is not None:
            img, target = self.transforms(img, target)
            return img, target
        elif self.transforms is not None:
            return self.transforms(img, None)[0]
        elif self.df_bbox is not None:
            return img, target
        else:
            return img

    def __iter__(self):
        self.iter_id = self.img_ids[0]
        return self

    def __next__(self):
        if self.iter_id == self.img_ids[1]-1:
            raise StopIteration
        return self.__getitem__(self.iter_id)

    def __len__(self):
        return len(self.imgs)


class UtilityModel:
    def __init__(self):
        pass

    @staticmethod
    def get_transform(train):
        transforms = [transf.ToTensor()]
        if train:
            transforms.append(transf.RandomHorizontalFlip(0.5))
        return transf.Compose(transforms)

    @staticmethod
    def get_model(num_classes):
        # load an object detection model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new on
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


class EarlyStopper():
    def __init__(self,
                 patience: int=5,
                 better: str="smaller",
                 best_value_init: float=50,
                 best_no_detection_init: int=50):
        """
        :param patience:
        :param better: 'bigger' or 'smaller'
        :param best_value_init: depending on better a value that is bigger or smaller than the worst expected
        validation result
        """
        self.patience = patience
        self.progress = {best_no_detection_init: [best_value_init]}
        self.best_model = None
        self.best_epoch = None
        self.no_detection = list()
        self.best_validation = best_value_init
        self.better = max if better=="bigger" else min
        self.margin = 0.9 if better=="bigger" else 1.1
        self.best_margin = self.best_validation * self.margin
        self.stopped_by = str()

    def consider_stopping(self, model, epoch: int, value: float or np.ndarray, no_detection: int) -> bool:
        stop = False
        self.no_detection.append(no_detection)
        if len(self.no_detection) == self.patience:
            stop = stop or self.__stop_no_detections()
            self.no_detection.pop(0)

        if not stop:
            best_no_detection = next(iter(self.progress))
            if no_detection < best_no_detection:
                self.progress = {no_detection: [value]}
                self.__update_best(value, model, epoch)
            elif no_detection == best_no_detection:
                if self.better(value, self.best_validation) == value:
                    self.__update_best(value, model, epoch)
                self.progress[best_no_detection].append(value)
                if len(self.progress[best_no_detection]) == self.patience:
                    stop = stop or self.__stop_value_progress()
                    self.progress[best_no_detection].pop(0)
        return stop

    def get_best_epoch(self) -> int:
        return self.best_epoch

    def get_best_model(self):
        return self.best_model

    def get_stopped_by(self):
        return self.stopped_by

    def __update_best(self, value, model, epoch):
        self.best_validation = value
        self.best_epoch = epoch
        self.best_model = model
        self.best_margin = self.best_validation * self.margin

    def __stop_no_detections(self) -> bool:
        consider_last = 4
        margin = 0
        margined_best_value = next(iter(self.progress))*(1+margin)
        return self.__inconsistent_progress(self.no_detection, margined_best_value, consider_last, min,
                                            "global_inconsistency")

    def __stop_value_progress(self) -> bool:
        must_progress_by = 0.03
        consider_last = 4
        values = list(self.progress.values())[0]
        slow_local_progress = self.__progress_too_slow(values,
                                                       must_progress_by,
                                                       self.better,
                                                       "local_slow_progress")
        local_inconsistent = self.__inconsistent_progress(values,
                                                          next(iter(self.progress))*self.margin,
                                                          consider_last,
                                                          self.better,
                                                          "local_inconsistency")

        return any([slow_local_progress, local_inconsistent])

    def __inconsistent_progress(self,
                                values: list,
                                margined_best_value: float,
                                include_last: int,
                                better,
                                criteria: str):
        """Stops if each of the last 'include_last' progress values was worse than the best validation value + a margin
        that occurred so far IF that best_value is not in the last 'include_last' progress values."""
        stop = list()
        progress_to_use = values[self.patience-include_last:]
        for value in progress_to_use:
            if better(value, margined_best_value) != value:
                stop.append(True)
                self.stopped_by = criteria
            else:
                stop.append(False)
        return all(i for i in stop)

    def __progress_too_slow(self,
                            values: list,
                            must_progress_by: float,
                            better,
                            criteria: str):
        """Stops the training if EACH (determined by 'patience') progress is smaller than must_progress_by."""
        stop = list()
        for i, value in enumerate(values[:-1]):
            if abs(1-better(values[i+1]/value, 1)) < must_progress_by:
                stop.append(True)
                self.stopped_by = criteria
            else:
                stop.append(False)
        return all(i for i in stop)


class TrainRig(UtilityModel, PredictionEvaluator):
    def __init__(self,
                 orders: prep.Orders,
                 root_dir: str):
        UtilityModel.__init__(self)
        PredictionEvaluator.__init__(self)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.orders = orders
        self.dir_root = root_dir
        self.test_rig = test.TestRig(self.orders, self.dir_root)

    def train_all(self,
                  num_classes: int=2,
                  batch_size: int=8,
                  print_progress: bool=False,
                  test_after_train: bool=False,
                  early_stopper_criteria: str=""):
        n_train_orders = self.orders.get_number_of_orders("train")
        for idx_train, dir_ids in enumerate(self.orders.get_all_parents_dir_idx("train")):
            if not test_after_train:
                helper.print_progress(idx_train, n_train_orders, "Training")
            else:
                helper.print_progress(idx_train, n_train_orders, "Training and testing")
            self.__train_dir_ids(dir_ids, num_classes, batch_size, print_progress, early_stopper_criteria)
            if test_after_train:
                test_dir_ids = self.orders.get_child_data_dirs({"test": dir_ids["train"]})
                self.test_rig.test_after_train(test_dir_ids, self.get_transform(False), num_classes)

    def __train_dir_ids(self,
                        dir_ids: dict,
                        num_classes: int,
                        batch_size: int,
                        print_progress: bool,
                        early_stopper_criteria: str,
                        validation_frequency: int=2):
        current_data_dir = self.dir_root + f"/data/{int(dir_ids['data'])}"
        dir_data_set = current_data_dir + "/imgs"
        file_bbox = current_data_dir + "/bbox.dat"
        dir_train = current_data_dir + f"/{dir_ids['train']}"

        if os.path.isdir(dir_train):
            if os.path.isfile(dir_train + f"/model.pt"):
                print(f"Already trained for these parameters (train_dir_idx: {dir_ids['train']})")
                return
            else:
                print(f"Unfinished training in (train_dir_idx: {dir_ids['train']}). Repeating that order.")
        else:
            helper.create_dir(dir_train)

        n_train = self.orders.get_value_for(dir_ids, "nTrainImgs")
        use_early_stopper = False
        if len(early_stopper_criteria) != 0:
            use_early_stopper = True
            early_stopper = EarlyStopper()
            validation_img_info = ast.literal_eval(self.orders.get_value_for(dir_ids, "valInfo"))
            val_data_sets = list()
            imgs_per_validation = validation_img_info[1]
            for data_set_idx in range(validation_img_info[0]):
                min_img_idx = n_train+data_set_idx*imgs_per_validation
                max_img_idx = n_train+(data_set_idx+1)*imgs_per_validation
                val_data_sets.append(CustomDataset(dir_data_set, file_bbox, (min_img_idx, max_img_idx),
                                                   self.get_transform(False)))
            data_set_val = itertools.cycle(val_data_sets)

        data_set = CustomDataset(dir_data_set, file_bbox, (0, n_train), self.get_transform(True))
        data_loader = DataLoader(data_set, batch_size=batch_size, collate_fn=collate_fn)
        model = self.get_model(num_classes=num_classes)
        model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        stopped_by = "max_epochs"
        for epoch in range(int(self.orders.get_value_for(dir_ids, "maxEpochs"))+1):
            train_one_epoch(model, optimizer, data_loader, self.device, epoch, print_freq=1,
                            print_progress=print_progress)
            lr_scheduler.step()

            if use_early_stopper and not epoch % validation_frequency:
                validation_value, no_detection = self.__validation(model, next(data_set_val), dir_ids,
                                                                   early_stopper_criteria)
                if validation_value is not None:
                    if early_stopper.consider_stopping(model, epoch, validation_value, no_detection):
                        model, epoch, stopped_by = early_stopper.get_best_model(), early_stopper.get_best_epoch(), \
                                                   early_stopper.get_stopped_by()
                        break
                else:
                    pass

        self.orders.set_secondary_parameter_values(self.orders.get_row_from_dir_ids(dir_ids, "train"),
                                                   epochs=epoch, stoppedBy=stopped_by)
        torch.save(model.state_dict(), dir_train + f"/model.pt")

    def __validation(self, model, data_set_val: CustomDataset, dir_ids: dict, early_stopper_criteria: str)->np.ndarray:
        df_predictions = self.test_rig.test_validation(model, data_set_val)
        file_bbox = self.dir_root + f"/data/{dir_ids['data']}/bbox.dat"
        self.set_predictions(df_predictions)
        self.set_truth(file_bbox)
        return self.get_criteria_mean(early_stopper_criteria)

