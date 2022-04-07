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
        """
        Images must be named 'number.png' in the directory and there must not be anything else in it.
        :param root:
        :param file_bbox:
        :param img_ids:
        :param transforms:
        """
        self.root = root
        self.df_bbox = pd.read_csv(file_bbox, index_col=None)
        self.transforms = transforms
        self.imgs = list()
        for id in range(len(list(os.listdir(self.root)))):
            self.imgs.append(f"{id}.png")
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
        if self.iter_id == self.img_ids[1]:
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
                 patience: int,
                 consider: int,
                 better: str,
                 n_dataset: int,
                 must_progress_by: float,
                 validation_frequency: int):
        """
        :param patience: epoch at which to start saving progress
        :param consider: how many epochs of each dataset to consider
        :param better: 'bigger' or 'smaller'
        :param n_datasets: number of different datasets used for the validation. They need to be cycled continuously
        :param must_progress_by: expected minimum percentage CHANGE per epoch
        validation result
        """
        self.patience = patience
        self.consider = consider
        self.n_dataset = n_dataset
        self.validation_frequency = validation_frequency
        self.best_model = None
        self.best_epoch = None
        self.best_validation = {idx_dataset: float() for idx_dataset in range(self.n_dataset)}
        self.best_margin = {idx_dataset: float() for idx_dataset in range(self.n_dataset)}
        self.progress = {idx_dataset: dict() for idx_dataset in range(self.n_dataset)}
        self.detected_normalised = {idx_dataset: list() for idx_dataset in range(self.n_dataset)}
        self.value_init = {idx_dataset: None for idx_dataset in range(self.n_dataset)}
        self.better = max if better=="bigger" else min
        self.margin = 0.9 if better=="bigger" else 1.1
        epochs_between_datasets = self.validation_frequency*(self.n_dataset-1)
        self.must_progress_by = abs(1-(self.better(1-must_progress_by, 1+must_progress_by))**(epochs_between_datasets))
        self.stopped_by = str()

    def consider_stopping(self, model, epoch: int, value: float or np.ndarray, detected: int or float) -> bool:
        idx_dataset = self.__get_idx_dataset(epoch)
        self.__update_dataset_progress(idx_dataset, model, epoch, value, detected)
        stop = []
        for idx_dataset in range(self.n_dataset):
            stop.append(self.__consider_dataset(idx_dataset))
        return all(stop)

    def get_best_epoch(self) -> int:
        return self.best_epoch

    def get_best_model(self):
        return self.best_model

    def get_stopped_by(self):
        return self.stopped_by

    def __get_idx_dataset(self, epoch):
        number_iteration_dataset = np.floor((epoch - self.patience) / (self.validation_frequency * self.n_dataset))
        number_dataset_wrong_freq = epoch-(self.patience+number_iteration_dataset*self.n_dataset*self.validation_frequency)
        number_dataset = number_dataset_wrong_freq/self.validation_frequency
        return int(number_dataset)

    def __update_dataset_progress(self, idx_dataset: int, model, epoch: int, value: float or np.ndarray,
                                  detected: int or float):
        dataset_detected = self.detected_normalised[idx_dataset]
        dataset_progress = self.progress[idx_dataset]
        best_validation = self.best_validation[idx_dataset]
        if value is None:
            if self.value_init[idx_dataset] is not None:
                value = self.value_init[idx_dataset]
        if epoch >= self.patience:
            if len(dataset_detected) == self.consider:
                dataset_detected.pop(0)
            dataset_detected.append(detected)
            if value is None:
                return
            if len(dataset_progress) == 0:
                dataset_progress[detected] = [value]
                self.value_init[idx_dataset] = value
                return
            best_detection = next(iter(dataset_progress))
            if detected > best_detection:
                dataset_progress[detected] = [value]
                dataset_progress.pop(best_detection)
                self.__update_best(idx_dataset ,value, model, epoch)
            elif detected == best_detection:
                if len(dataset_progress[detected]) == self.consider:
                    dataset_progress[detected].pop(0)
                dataset_progress[detected].append(value)
                if self.better(value, best_validation) == value:
                    self.__update_best(idx_dataset ,value, model, epoch)
        return

    def __consider_dataset(self, idx_dataset: int) -> bool:
        stop = False
        detected_normalised = self.detected_normalised[idx_dataset]
        if len(detected_normalised) != self.consider:
            return stop
        progress = list(self.progress[idx_dataset].values())[0]
        stop = self.__stop_detections(detected_normalised, idx_dataset)
        if len(progress) == self.consider:
            stop = stop or self.__stop_value_progress(progress, idx_dataset)
        return stop

    def __update_best(self, idx_dataset: int, value: float, model, epoch: int):
        self.best_validation[idx_dataset] = value
        self.best_epoch = epoch
        self.best_model = model
        self.best_margin[idx_dataset] = self.best_validation[idx_dataset] * self.margin

    def __stop_detections(self, detected_normalised: list, idx_dataset: int) -> bool:
        margined_best_value = next(iter(self.progress[idx_dataset]))*0.9
        return self.__inconsistent_progress(detected_normalised, margined_best_value, max, "global_inconsistency")

    def __stop_value_progress(self, progress: list, idx_dataset: int) -> bool:
        slow_local_progress = self.__progress_too_slow(progress,
                                                       self.must_progress_by,
                                                       self.better,
                                                       "local_slow_progress")
        dataset_best = self.best_margin[idx_dataset]
        local_inconsistent = self.__inconsistent_progress(progress,
                                                          dataset_best,
                                                          self.better,
                                                          "local_inconsistency")
        return any([slow_local_progress, local_inconsistent])

    def __inconsistent_progress(self,
                                progress: list,
                                margined_best_value: float,
                                better,
                                criteria: str):
        """Stops if each of the last 'consider' progress values was worse than the best validation value + a margin
        that occurred so far."""
        stop = list()
        for value in progress:
            if better(value, margined_best_value) == margined_best_value:
                stop.append(True)
                self.stopped_by = criteria
            else:
                stop.append(False)
        return all(stop)

    def __progress_too_slow(self,
                            progress: list,
                            must_progress_by: float,
                            better,
                            criteria: str):
        """Stops the training if EACH (determined by 'consider') progress is smaller than must_progress_by."""
        stop = list()
        for i, value in enumerate(progress[:-1]):
            next_value = progress[i+1]
            if abs(1-better(next_value/value, 1)) < must_progress_by:
                stop.append(True)
                self.stopped_by = criteria
            else:
                stop.append(False)
        return all(stop)


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

    def all(self,
            num_classes: int=2,
            batch_size: int=8,
            print_progress: bool=False,
            test_after_train: bool=False):
        n_train_orders = self.orders.get_number_of_orders("train")
        for idx_train, dir_ids in enumerate(self.orders.get_all_parents_dir_idx("train")):
            if not test_after_train:
                helper.print_progress(idx_train, n_train_orders, "Training")
            else:
                helper.print_progress(idx_train, n_train_orders, "Training and testing")
            self.__train_dir_ids(dir_ids, num_classes, batch_size, print_progress)
            if test_after_train:
                test_dir_ids = self.orders.get_child_data_dirs({"test": dir_ids["train"]})
                self.test_rig.test_after_train(test_dir_ids, self.get_transform(False), num_classes)

    def __train_dir_ids(self,
                        dir_ids: dict,
                        num_classes: int,
                        batch_size: int,
                        print_progress: bool,
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

        n_train = self.orders.get_value_for(dir_ids, "nImgTrain")
        early_stopper_criteria = self.orders.get_value_for(dir_ids, "earlyStopperCriteria")
        if early_stopper_criteria is not None:
            validation_info = ast.literal_eval(self.orders.get_value_for(dir_ids, "validationInfo"))
            val_data_sets = list()
            imgs_per_validation = validation_info[1]
            for data_set_idx in range(validation_info[0]):
                min_img_idx = n_train+data_set_idx*imgs_per_validation
                max_img_idx = n_train+(data_set_idx+1)*imgs_per_validation
                val_data_sets.append(CustomDataset(dir_data_set, file_bbox, (min_img_idx, max_img_idx),
                                                   self.get_transform(False)))
            data_set_val = itertools.cycle(val_data_sets)

            assign_criteria = self.orders.get_value_for(dir_ids, "assignCriteria")
            assign_threshold = self.orders.get_value_for(dir_ids, "assignTpThreshold")
            need_size = True if assign_criteria == "distance" else False
            if need_size:
                assign_threshold *= self.__get_img_size(dir_ids)[0] # distance is not absolut and not normalised
            self.set_assignment(criteria=assign_criteria,
                                better=self.orders.get_value_for(dir_ids, "assignBetter"),
                                threshold=assign_threshold)

            self.set_prediction_criteria(criteria=self.orders.get_value_for(dir_ids, "predictionCriteria"),
                                         better=self.orders.get_value_for(dir_ids, "predictionBetter"),
                                         threshold=self.orders.get_value_for(dir_ids, "predictionThreshold"))
            consider = 2
            val_info = self.orders.get_value_for(dir_ids, "validationInfo")
            n_dataset = helper.str_list2value(val_info, 0)
            better = "bigger" if early_stopper_criteria == "ap" else "smaller" # else means distance
            early_stopper = EarlyStopper(patience=0, consider=consider, better=better, n_dataset=n_dataset,
                                         must_progress_by=0.05, validation_frequency=validation_frequency)

            file_additional_information = current_data_dir + "/imgsInformation.dat"
            df_additional_info = pd.read_csv(file_additional_information, index_col=None)
            df_n_vortices = df_additional_info["n_vortices"]


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

            if early_stopper_criteria is not None and not epoch % validation_frequency:
                next_data_set = next(data_set_val)
                validation_value, n_detected_vortices = self.__validation(model, next_data_set, dir_ids,
                                                                          early_stopper_criteria)
                current_img_ids = next_data_set.img_ids
                n_actual_vortices = df_n_vortices[current_img_ids[0]:current_img_ids[1]].sum()
                detected_normalised = n_detected_vortices/n_actual_vortices
                if early_stopper.consider_stopping(model, epoch, validation_value, detected_normalised):
                    model, epoch, stopped_by = early_stopper.get_best_model(), early_stopper.get_best_epoch(), \
                                               early_stopper.get_stopped_by()
                    break
                else:
                    pass
        self.orders.set_secondary_parameter_values(self.orders.get_row_from_dir_ids(dir_ids, "train"),
                                                   epochs=epoch, stoppedBy=stopped_by)
        torch.save(model.state_dict(), dir_train + f"/model.pt")

    def __validation(self, model, data_set_val: CustomDataset, dir_ids: dict, early_stopper_criteria: str)\
            -> tuple[np.ndarray, int]:
        df_predictions = self.test_rig.test_validation(model, data_set_val)
        file_bbox = self.dir_root + f"/data/{dir_ids['data']}/bbox.dat"
        self.set_predictions(df_predictions)
        self.set_truth(file_bbox)
        criteria_mean, n_detected_vortices = self.get([early_stopper_criteria])
        return criteria_mean[early_stopper_criteria], n_detected_vortices

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