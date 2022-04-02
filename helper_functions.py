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
    def df_predictions_to_list(df: pd.DataFrame) -> list[tuple[float, list]]:
        translated = list()
        for _, row in df.iterrows():
            try:
                translated.append((row["score"], ast.literal_eval(row["bbox"])))
            except ValueError:
                translated.append((row["score"], row["bbox"]))
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
    def save_parameters(working_dir: str, **kwargs):
        file = working_dir + "/parameters.dat"
        try:
            df_parameters = pd.read_csv(file, index_col=None)
            last_set = df_parameters["index_experiment"].max()
        except FileNotFoundError:
            df_parameters = pd.DataFrame()
            last_set = 0
        df_new_set = pd.DataFrame({param: pd.Series(values) for param, values in kwargs.items()}, index=None)
        df_new_set["index_experiment"] = [last_set + 1 for _ in range(df_new_set.shape[0])]
        df_parameters = pd.concat([df_parameters, df_new_set], ignore_index=True)
        df_parameters.to_csv(file, index=False)


def create_directory(
        directory: str,
        overwrite: bool = False
) -> str:
    path_dir = directory
    if os.path.isdir(directory):
        if overwrite:
            shutil.rmtree(path_dir)
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Existing directory {path_dir} was deleted and a new one created.")
        else:
            print(f"Directory {directory} already exists and was not overwritten.")
    else:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory {path_dir} was created.")
    return path_dir


def create_directory_from_dict(
        cwd: str,
        variable: dict,
        overwrite: bool = False,
        create: bool = True
) -> str:
    path_dir = cwd
    for variable_name, variable_value in variable.items():
        if variable_value is not None:
            path_dir += f"/{variable_name}_{variable_value}"
        else:
            path_dir += f"/{variable_name}"
    if os.path.isdir(path_dir):
        if overwrite and create:
            shutil.rmtree(path_dir)
            print(f"Directory {path_dir} has been removed.")
            pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)
            print(f"Directory {path_dir} has been created.")
        else:
            print(f"Directory {path_dir} already exists and was not overwritten.")
    elif create:
        pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)
        print(f"Directory {path_dir} has been created.")
    return path_dir


def version_from_string(string: str) -> int:
    dot_id = string.find(".")
    is_number = True
    plot_no = str()
    prior_id = 1
    while is_number:
        next_number = string[dot_id - prior_id]
        try:
            int(next_number)
            plot_no += next_number
            prior_id += 1
        except ValueError:
            is_number = False
    return int(plot_no[::-1])


def get_save_version(
        dir2check: str,
        file_type2check: str,
        file_name_contains: str
) -> int:
    type_files = list()
    for file in [type_files for type_files in
                 glob.glob(dir2check + f"/*.{file_type2check}") if file_name_contains in type_files]:
        type_files.append(file)
    max_no = 0
    for file in type_files:
        file_no = version_from_string(file)
        if file_no > max_no:
            max_no = file_no
    return max_no + 1


def append_to_file(file: str, to_append: str) -> str:
    dot_id = file.find(".")
    return file[:dot_id] + to_append + file[dot_id:]





def get_value_from_path(path: str, value_for: str, param_value_sep: str = "_") -> int or float or list:
    idx_param_start = path.find(value_for)
    param_and_value_and_appending = path[idx_param_start:]
    param_and_value = param_and_value_and_appending[:param_and_value_and_appending.find("/")]
    value = param_and_value[param_and_value.find(param_value_sep) + 1:]

    try:
        value = float(value)
    except ValueError:
        value = ast.literal_eval(value)
        return value
    try:
        if int(value) == value:
            value = int(value)
    except ValueError:
        pass
    return value


def print_train_test_progress(train_or_test: dict, idx_current_order: int, total_orders: int) -> None:
    msg = f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: "
    if train_or_test["train"]:
        msg += "Training "
        if train_or_test["test"]:
            msg += "and testing "
    elif train_or_test["test"]:
        msg += "Testing "
    print(msg + f"on data order {idx_current_order} of {total_orders} -> {idx_current_order / total_orders * 100}%")


def new_path(
        parent_dir_path: str,
        var_name: str,
        var_value,
        create: bool = True,
        override: bool = False,
        new_case: bool = False
) -> str:
    accepted_types = [int, float, list, str]
    if parent_dir_path == "":
        cwd_new = var_name
        if var_value is not None:
            cwd_new += f"{var_value}"
    elif var_value is None:
        new_dir_name = var_name
        cwd_new = parent_dir_path + "/" + new_dir_name
    elif type(var_value) in accepted_types:
        new_dir_name = var_name + f"{var_value}"
        cwd_new = parent_dir_path + "/" + new_dir_name
    else:
        raise ValueError(f"var_value must be of type {accepted_types} or None but was {type(var_value)}.")
    if create:
        if os.path.isdir(cwd_new):
            if override:
                os.remove(cwd_new)
                os.mkdir(cwd_new)
                print(f"{cwd_new} has been overridden.")
            elif new_case:
                _, dirs, _ = next(os.walk(parent_dir_path))
                previous_version_dir = []
                for directory in dirs:
                    if new_dir_name in directory:
                        previous_version_dir.append(directory)
                previous_dir = previous_version_dir[-1]
                idx_number_begin = [index + 1 for index, character in enumerate(previous_dir) if character == "_"]
                if len(idx_number_begin) < 1:
                    cwd_new += "_1"
                else:
                    last_test_id = int(previous_dir[idx_number_begin[-1]:])
                    cwd_new += f"_{last_test_id + 1}"

                try:
                    os.mkdir(cwd_new)
                    print(f"{cwd_new} has been created.")
                except OSError:
                    raise OSError(f"Creation of the directory {cwd_new} failed. Aborting.")
            else:
                pass
        else:
            try:
                os.mkdir(cwd_new)
                print(f"{cwd_new} has been created.")
            except OSError:
                raise OSError(f"Creation of the directory {cwd_new} failed. Aborting.")
    return cwd_new


def get_extreme_value_csv(
        data_frame: DataFrame,
        column_main_crit: str,
        get_max_main: bool,
        column_backup_crit: str,
        get_max_backup: bool
) -> float:
    df_size = data_frame.shape
    if df_size[0] < 2:
        return data_frame.index.tolist()[0]

    if get_max_main:
        idx_extreme = data_frame[column_main_crit].idxmax()
    else:
        idx_extreme = data_frame[column_main_crit].idxmin()

    there_is_more = True
    while there_is_more:
        temp_df = data_frame.drop(idx_extreme)
        if get_max_main:
            idx_next_extreme = temp_df[column_main_crit].idxmax()
        else:
            idx_next_extreme = temp_df[column_main_crit].idxmin()

        if data_frame[column_main_crit][idx_extreme] == temp_df[column_main_crit][idx_next_extreme]:
            if get_max_backup:
                if data_frame[column_backup_crit][idx_extreme] < temp_df[column_backup_crit][idx_next_extreme]:
                    idx_extreme = idx_next_extreme
                else:
                    there_is_more = False
            else:
                if data_frame[column_backup_crit][idx_extreme] > temp_df[column_backup_crit][idx_next_extreme]:
                    idx_extreme = idx_next_extreme
                else:
                    there_is_more = False
        else:
            there_is_more = False
    return idx_extreme
