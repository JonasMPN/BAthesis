import pathlib
import os
from typing import List, Tuple
import shutil
import glob
import ast
import datetime

import pandas as pd
import torch
from pandas import DataFrame
import numpy as np


class Helper():
    def __init__(self):
        pass

    def create_dir(self,
                   path_dir: str,
                   overwrite: bool = False,
                   add_missing_parent_dirs: bool = True,
                   raise_exception: bool = False,
                   print_message: bool = False) \
            -> tuple[str, bool]:
        return self.__create_dir(path_dir, overwrite, add_missing_parent_dirs, raise_exception, print_message)

    def create_file(self,
                    path_file: str,
                    overwrite: bool = False,
                    add_missing_parent_dirs: bool = True,
                    raise_exception: bool = False,
                    print_message: bool = False) \
            -> tuple[str, bool]:
        dir_file = path_file[::-1][path_file[::-1].find("/") + 1:][::-1]
        self.__create_dir(dir_file, False, add_missing_parent_dirs, raise_exception, False)
        try:
            open(path_file, "x")
            msg, continue_mission = f"File {path_file} was successfully created.", True
        except FileExistsError:
            if raise_exception:
                raise FileExistsError(f"File {path_file} already exists.")
            if overwrite:
                pathlib.Path(path_file).unlink()
                tmp = open(path_file, "x")
                tmp.close()
                msg, continue_mission = f"File {path_file} overwritten.", True
            else:
                msg, continue_mission = f"File {path_file} already exists.", False
        if print_message:
            print(msg)
        return msg, continue_mission

    @staticmethod
    def __create_dir(target: str,
                     overwrite: bool,
                     add_missing_parent_dirs: bool,
                     raise_exception: bool,
                     print_message: bool) \
            -> tuple[str, bool]:
        msg, keep_going = str(), bool()
        try:
            if overwrite:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    msg = f"Existing directory {target} was overwritten."
                else:
                    msg = f"Could not overwrite {target} as it did not exist. Created it instead."
                keep_going = True
            else:
                msg, keep_going = f"Directory {target} created successfully.", True
            pathlib.Path(target).mkdir(parents=add_missing_parent_dirs, exist_ok=False)
        except Exception as exc:
            if exc.args[0] == 2:  # FileNotFoundError
                if raise_exception:
                    raise FileNotFoundError(f"Not all parent directories exist for directory {target}.")
                else:
                    msg, keep_going = f"Not all parent directories exist for directory {target}.", False
            elif exc.args[0] == 17:  # FileExistsError
                if raise_exception:
                    raise FileExistsError(f"Directory {target} already exists.")
                else:
                    msg, keep_going = f"Directory {target} already exists.", False
        if print_message:
            print(msg)
        return msg, keep_going

    def print_progress(self,
                       idx_current_order: int,
                       total_orders: int,
                       progressing_on: str) -> None:
        msg = f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {progressing_on} "
        print(msg + f"on order {idx_current_order} of {total_orders} -> "
                    f"{np.round(idx_current_order / total_orders * 100,2)}%")

    def warning(self, text: str):
        print(self.coloured_text(text, r=255, g=100, b=100))

    @staticmethod
    def coloured_text(text: str, colour: str = None, r: int = None, g: int = None, b: int = None):
        if colour is not None:  # dismiss rgb values
            implemented_colours = {
                "red": (255, 0, 0)
            }
            r, g, b = implemented_colours[colour]
        return f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m"

    @staticmethod
    def pd_row2(
            dataframe: DataFrame,
            index_col: int,
            to="torch" or "list" or "draw_tuple",
    ) -> Tuple[torch.Tensor or List[List[float]], int]:
        boxes_list = dataframe.loc[index_col].tolist()  # cols in the df are xmin, ymin, xmax, ymax
        x_min_values = ast.literal_eval(boxes_list[0])
        y_min_values = ast.literal_eval(boxes_list[1])
        x_max_values = ast.literal_eval(boxes_list[2])
        y_max_values = ast.literal_eval(boxes_list[3])
        num_objs = len(x_min_values)
        if to == "torch":
            boxes = torch.zeros([num_objs, 4])
            for box in range(num_objs):
                boxes[box, 0] = x_min_values[box]
                boxes[box, 1] = y_min_values[box]
                boxes[box, 2] = x_max_values[box]
                boxes[box, 3] = y_max_values[box]
            return boxes, num_objs
        if to == "list":
            boxes = list()
            for box in range(num_objs):
                boxes.append([x_min_values[box], y_min_values[box], x_max_values[box], y_max_values[box]])
            return boxes, num_objs
        if to == "draw_tuple":
            boxes = list()
            for box in range(num_objs):
                boxes.append([(x_min_values[box], y_min_values[box]), (x_max_values[box], y_max_values[box])])
            return boxes, num_objs

    @staticmethod
    def path_from_order(
            root: str,
            variable: dict
    ) -> str:
        path = root
        for variable_name, variable_value in variable.items():
            if variable_value is not None:
                path += f"/{variable_name}_{variable_value}"
            else:
                path += f"/{variable_name}"
        return path

    @staticmethod  # todo change the directory names to 'order_type'_'idx'
    def path_from_dir_ids(
            root: str,
            dir_ids: dict
    ) -> str:
        path = root
        for idx in dir_ids.values():
            path += f"/{idx}"
        return path

    @staticmethod
    def df_predictions_to_list(df: pd.DataFrame, tp_criteria:str) -> list[tuple[float, list]]:
        translated = list()
        for _, row in df.iterrows():
            try:
                translated.append((row[tp_criteria], ast.literal_eval(row["bbox"])))
            except ValueError:
                translated.append((row[tp_criteria], row["bbox"]))
        return translated

    @staticmethod
    def interval_to_shape_params(interval: list) -> tuple:
        stretch = interval[1] - interval[0]
        offset = interval[0]
        return stretch, offset

    @staticmethod
    def merge_list_of_dicts(dicts: List[dict]) -> dict:
        keys = list(dicts[0].keys())
        combined = {key: list() for key in keys}
        for dic in dicts:
            for key in keys:
                combined[key].append(dic[key])
        return combined

    @staticmethod
    def df_to_orders(dataframe: pd.DataFrame) -> list[dict]:
        orders = list()
        for _, row in dataframe.iterrows():
            orders.append(row.to_dict())
        return orders

    @staticmethod
    def df_row_to_dict(series: pd.Series) -> dict:
        row_dict = dict()
        for param, value in series.items():
            row_dict[param] = value[0]
        return row_dict

    @staticmethod
    def str_list2value(str, idx):
        return ast.literal_eval(str)[idx]

    @staticmethod
    def save_parameters(working_dir: str, parameters: dict) -> tuple[dict, int]:
        paths = dict()
        for order_type in parameters.keys():
            file = working_dir + f"/parameters_{order_type}.dat"
            try:
                df_parameters = pd.read_csv(file, index_col=None)
                last_set = df_parameters["indexExperiment"].max()
            except FileNotFoundError:
                df_parameters = pd.DataFrame()
                last_set = 0
            df_new_set = pd.DataFrame({param: pd.Series(values) for param, values in parameters[order_type].items()},
                                      index=None)
            df_new_set["indexExperiment"] = [last_set + 1 for _ in range(df_new_set.shape[0])]
            df_parameters = pd.concat([df_parameters, df_new_set], ignore_index=True)
            df_parameters.to_csv(file, index=False)
            paths[order_type] = file
        return paths, last_set+1
