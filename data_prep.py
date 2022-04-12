import itertools
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import mean_squared_error as rms
import scipy.io as io

from helper_functions import Helper
from PIL import Image, ImageDraw
import os, ast
from copy import copy

helper = Helper()


class Orders:
    """
    order: An order consists of only the parameters that are set for the instance of the class Orders.
    order_type: Every parameter has to be allocated to one order_type, e.g. number_images and size_images to the
                order type data
    order_type_succession.
    It is currently not supported to set the parameters to new parameters without reinitialising the instance of the
    object (unwanted behaviour possible).
    """
    def __init__(self,
                 order_type_succession: list[str],
                 working_dir: str):
        """The database files have their root in 'working_dir' and will be named according to
        'order_type_succession'. All existing order types must be supplied."""
        self.order_type_succession = order_type_succession
        self.working_dir = working_dir
        helper.create_dir(self.working_dir)
        self.file_databases = dict()
        self.df_databases = dict()
        self.existing_parameters = dict()

        for order_type in self.order_type_succession:
            self.file_databases[order_type] = working_dir + "/" + f"database_{order_type}.dat"
            try:
                self.df_databases[order_type] = pd.read_csv(self.file_databases[order_type], index_col=None)
                columns = self.df_databases[order_type].columns.tolist()
                self.existing_parameters[order_type] = [param for param in columns if "dir" not in param]
            except FileNotFoundError:
                self.df_databases[order_type] = None
                self.existing_parameters[order_type] = None
        self.primary_parameters = dict()
        self.secondary_parameters = dict()
        self.orders = dict()
        self._full_orders = dict()
        self._df_dir_ids = pd.DataFrame()
        self.__ignore_param_for_order = dict()

    def set_params(self, order_type: str, secondary_parameters: list = None, **kwargs) -> None:
        """The first input always has to be the name of the order type, being passed not as an kwarg"""
        secondary_parameters = secondary_parameters if secondary_parameters is not None else []
        if self.df_databases[order_type] is not None:
            if any(param not in secondary_parameters + list(kwargs.keys()) for param in self.existing_parameters[
                order_type]):
                helper.warning(f"The {order_type} order is missing at least one parameter that already "
                               f"exists in {self.file_databases[order_type]}. Missing parameter values will be "
                               f"filled with 'None' (str).")
            if any(param not in self.existing_parameters[order_type] for param in secondary_parameters + list(
                    kwargs.keys())):
                helper.warning(f"A new parameter has been added to the database {self.file_databases[order_type]}. "
                               f"Previous {order_type} order will have a 'None' (str) for that parameter.")
        self.primary_parameters[order_type] = kwargs
        self.secondary_parameters[order_type] = {param: "None" for param in secondary_parameters}
        if self.df_databases[order_type] is None:
            self.df_databases[order_type] = pd.DataFrame(columns=list(kwargs.keys())+secondary_parameters)

    def set_params_from_files(self, files: dict, index_experiment: int):
        for order_type, file in files.items():
            df_params = pd.read_csv(file, index_col=None)
            df_params = df_params.loc[df_params["indexExperiment"] == index_experiment]
            self.__set_params_from_file(df_params, order_type)

    def set_ignore_params_for_order(self, order_type: str, ignore_dict: dict) -> None:
        """ignore_dict example: {param: [trigger_value, ignore_param1, ignore_param2,...]}"""
        self.__ignore_param_for_order[order_type] = ignore_dict

    def set_secondary_parameter_values(self, row: dict, **kwargs):
        """Row w/o the secondary parameters"""
        order_type = self.__get_order_types(row, dismiss_secondary=True)[0]
        for secondary_parameter in kwargs:
            if secondary_parameter not in self.secondary_parameters[order_type]:
                raise ValueError(f"Existing secondary parameter values for order type {order_type} are "
                                 f"{self.secondary_parameters[order_type]} but parameter {secondary_parameter} was "
                                 f"tried to be set.")
            else:
                query = self.__order_to_query(row)
                index = self.df_databases[order_type].query(expr=query).index[0]
                self.df_databases[order_type].at[index, secondary_parameter] = kwargs[secondary_parameter]
        self.df_databases[order_type].to_csv(self.file_databases[order_type], index=False)

    def get_database_files(self) -> dict:
        return copy(self.file_databases)

    def get_database_dataframes(self) -> tuple:
        dataframes = list()
        for df in list(self.df_databases.values()):
            dataframes.append(df)
        return tuple(dataframes)

    def get_value_for(self, parents_dir_ids: dict, param: str):
        order_type = self.__match_param_order_type(param)
        return self.df_databases[order_type].loc[parents_dir_ids[order_type]][param]

    def get_row_from_dir_ids(self, parents_dir_ids: dict, row_order_type: str, ignore_secondary: bool=True) -> dict:
        if ignore_secondary:
            return self.__primary(row_order_type).loc[parents_dir_ids[row_order_type]].to_dict()
        else:
            return self.df_databases[row_order_type].loc[parents_dir_ids[row_order_type]].to_dict()

    def get_dir_id(self, row) -> int:
        order_type = self.__get_order_types(row)[0]
        query = self.__order_to_query(row)
        return self.df_databases[order_type].query(expr=query).index[0]

    def get_number_of_orders(self, order_type: str) -> int:
        return len(self._full_orders[order_type])

    def get(self, order_type: str, return_type: str = "dict") -> list[dict] or list[str]:
        """Returns all the orders of an order_type. This function is meant to be used in combination with an iter
        statement. If done so, every yielded object of the iterator is of type 'return_type'."""
        accepted_order_types = ["data", "train", "test"]
        if order_type not in accepted_order_types:
            raise AttributeError(f"'order_type' must be one of {accepted_order_types} but was {order_type}.")
        if return_type == "dict":
            return self.orders[order_type]
        elif return_type == "query":
            query_return = list()
            for order in self.orders[order_type]:
                query_return.append(self.__order_to_query(order))
            return query_return

    def get_full_params_from_dir_ids(self, parent_dir_ids: dict) -> dict:
        params = dict()
        for order_type, dir_id in parent_dir_ids.items():
            params = params|self.df_databases[order_type].loc[dir_id].to_dict()
        return params

    def get_full_params_from_database_row(self, row: dict):
        """Returns all parameters that are defined for the parameter combination of the row. This includes all the
        parent order type parameters."""
        order_types = self.__get_order_types(row)
        if order_types[0] == self.order_type_succession[0]:
            query = self.__order_to_query(row)
            return self.df_databases[order_types[0]].query(expr=query).to_dict()
        parent_order_types = self.order_type_succession[:self.order_type_succession.index(order_types[0])]
        parent_order_types.reverse()
        params = copy(row)
        for parent_order_type in parent_order_types:
            params = (self.df_databases[parent_order_type].iloc[int(params[f"dir_{parent_order_type}"])].to_dict() |
                      params)
            params.pop(f"dir_{parent_order_type}")
        return params

    def get_child_data_dirs(self, specifier: dict) -> list[dict]:
        """Solution to the following problem: 5 order types exist. You've got ONE combination for the first 2 order
        types (parent_data_dirs) and want to get all the dir combinations (for the current set of parameters) of
        the remaining 3 order types (child_data_dirs) that stem from the parent_data_dir combination. Every returned
        dict contains the parent_data_dirs.
        specifier: Either {order_type: dir_id} or {order_type: dir_id} with additional parent order types and their
        dir_ids as values. In the second case the lowest order type in the inheritance order will be chosen."""
        if self._df_dir_ids.empty:
            self.__create_dir_ids()
        if len(list(specifier.keys())) > 1:
            raise NotImplementedError("Add functionality to support the second case.")
        df_filtered = self._df_dir_ids.loc[self._df_dir_ids[list(specifier.keys())[0]] == list(specifier.values())[0]]
        return self.__df_to_list_of_dicts(df_filtered)

    def get_all_parents_dir_idx(self, order_type: str) -> list[dict]:
        """Returns a list containing dictionaries. The parent dirs are the keys and their indices the value."""
        if self._df_dir_ids.empty:
            self.__create_dir_ids()
        child_order_types = self.order_type_succession[self.order_type_succession.index(order_type)+1:]
        df_filtered = self._df_dir_ids.drop(child_order_types, axis=1)
        df_filtered = df_filtered.drop_duplicates(subset=order_type)
        return self.__df_to_list_of_dicts(df_filtered)

    def get_parent_dir_idx(self, child_id: dict) -> dict:
        """
        Solves the task: 5 order_types, in the data_base for the 4th there is a row with dir_{3rd order_type}=x. Now
        you want to know which dir_ids (from the 1st and 2nd order_type) belong to that row. This function is
        independent from the current set of parameters.
        :param child_id: {{first_parent_order_type}: {dir_id}
        :return: dir_order_types as keys and their indices as values
        """
        last_order_type = next(iter(child_id))
        prior_order_types = self.__get_parent_order_types(last_order_type)[::-1]
        parent_dir_idx = child_id
        dir_id = list(child_id.values())[0]
        for id, order_type in enumerate([last_order_type]+prior_order_types[:-1]):
            df_parent = self.df_databases[order_type]
            dir_id = df_parent[f"dir_{prior_order_types[id]}"][dir_id]
            parent_dir_idx[prior_order_types[id]] = int(dir_id)
        return parent_dir_idx

    def get_params(self, order_type, primary: bool=True, secondary: bool=True) -> list[str]:
        params = list()
        if primary:
            params += self.primary_parameters[order_type]
        if secondary:
            params += self.secondary_parameters[order_type]
        return params

    def get_all_params(self):
        params = list()
        for order_type in self.order_type_succession:
            params += self.get_params(order_type)
        return params

    def create_orders(self, order_algorithm: str) -> None:
        for order_type in self.order_type_succession:
            self.__create_order(order_type, order_algorithm)
        return None

    def create_order(self, order_type: str, order_algorithm: str) -> None:
        self.__create_order(order_type, order_algorithm)
        return None

    def conduct_orders(self):
        for order_type in self.order_type_succession:
            self.__conduct_order_for_type(order_type)

    def conduct_order(self, order_type):
        self.__conduct_order_for_type(order_type)

    def __create_order(self, order_type: str, order_algorithm: str):
        """Does not contain secondary parameters"""
        implemented = {"full_fac": self.__full_fac, "specific": self.__specific}
        if order_algorithm not in list(implemented.keys()):
            raise NotImplementedError(f"order_algorithm '{order_type}' is not implemented. Implemented algorithms are "
                                      f"{[*implemented]}.")
        accepted_order_types = self.order_type_succession
        if order_type not in accepted_order_types:
            raise AttributeError(f"'order_type' must be one of {accepted_order_types} but was {order_type}.")
        self.orders[order_type] = implemented[order_algorithm](order_type)
        self.__clean_orders(order_type)

    def __create_dir_ids(self):
        """Creates all directory index combinations that are valid for the current set of parameters. Does not take
        the secondary parameters into account."""
        last_order_type = self.order_type_succession[-1]
        row_ids_order_type = self.__get_row_ids(last_order_type)
        for row_id in row_ids_order_type:
            df_filtered_primary = self.__primary(last_order_type).loc[row_id]
            self._df_dir_ids = self._df_dir_ids.append(self.__get_parent_idx(df_filtered_primary), ignore_index=True)

    def __conduct_order_for_type(self, order_type: str):
        """"""
        if self.df_databases[order_type] is None:
            cols = list(self.primary_parameters[order_type].keys()) + list(self.secondary_parameters[order_type].keys())
            self.df_databases[order_type] = pd.DataFrame(columns=cols)
            self.df_databases[order_type].to_csv(self.file_databases[order_type], index=False)
        else:
            pass

        orders = copy(self.orders[order_type])
        if order_type is not self.order_type_succession[0]:
            parented_orders = orders
            parent_param_type = self.order_type_succession[self.order_type_succession.index(order_type) - 1]
            orders = list()
            prior = self.__get_prior_order_type(order_type)
            parent_dir_idx = self.__get_row_ids(prior)
            for parent_dir_id in parent_dir_idx:
                for order in parented_orders:
                    orders.append({f"dir_{parent_param_type}": parent_dir_id} | order)
        self._full_orders[order_type] = orders
        for col in list(orders[0].keys()):
            if col not in self.df_databases[order_type].columns.tolist():
                self.df_databases[order_type].insert(loc=0, column=col, value=None)

        df_tmp = self.__primary(order_type)
        new_orders = list()
        for order in orders:
            if df_tmp.query(expr=self.__order_to_query(order)).empty:
                df_tmp = df_tmp.append(order, ignore_index=True)
                new_orders.append(order)
        if len(new_orders) != 0:
            df_new_orders = pd.DataFrame()
            new_orders = helper.merge_list_of_dicts(new_orders)
            for param, values in new_orders.items():
                if type(values[0]) == int:
                    param_col = pd.Series(values, dtype="int32")
                else:
                    param_col = pd.Series(values)
                df_new_orders[param] = param_col
            self.df_databases[order_type] = self.df_databases[order_type].append(df_new_orders, ignore_index=True)
            self.df_databases[order_type] = self.df_databases[order_type].replace(np.nan, "None")
            self.df_databases[order_type].to_csv(self.file_databases[order_type], index=False)

    def __get_order_types(self, multi_order: dict, dismiss_secondary: bool=False) -> list[str]:
        """Returns all order types that are explicitly existing in a multidimensional order,that is, an order that
        contains multiple order types. Also works for ordinary orders and orders that are a row from a database (
        therefore containing a 'dir_parent_order_type' parameter)."""
        tmp_order = copy(multi_order)
        for param in list(tmp_order.keys()):
            if "dir" in param:
                tmp_order.pop(param)
        order_types = list()

        for order_type in self.order_type_succession:
            if dismiss_secondary:
                tmp_df = self.__primary(order_type)
                necessary_parameters = [param for param in tmp_df.columns.tolist()  if "dir" not in param]
            else:
                necessary_parameters = [param for param in self.df_databases[order_type].columns.tolist()  if "dir" not
                                        in param]
            supplied_params = list(tmp_order.keys())
            if any(param in supplied_params for param in necessary_parameters):
                order_types.append(order_type)
                for param in necessary_parameters:
                    try:
                        tmp_order.pop(param)
                    except KeyError:
                        raise KeyError(f"Missing parameter {param} for order type {order_type}. Required parameters "
                                       f"for that order type are {necessary_parameters} but supplied were only "
                                       f"{supplied_params}.")
        order_without_dirs = {param: value for param, value in multi_order.items() if "dir" not in param}
        self.__validate_values_of_order(order_without_dirs, order_types, dismiss_secondary)
        return order_types

    def __get_row_ids(self, order_type: str) -> list[int]:
        """This function is a solution to the problem 'Which rows of the database for the order_type are part of the
        current set of parameters (which have been set with the set_params method)."""
        parent_order_types = self.__get_parent_order_types(order_type)
        tmp_df = self.__primary(self.order_type_succession[0])
        row_idx = list()
        for parent_order_type in parent_order_types+[order_type]:
            row_idx = list()
            for order in self.orders[parent_order_type]:
                query = self.__order_to_query(order)
                row_idx += tmp_df.query(expr=query).index.tolist()
            if parent_order_type == order_type:
                break

            keep_idx = list()
            next_parent_order_type = self.__get_next_order_type(parent_order_type, return_on_overflow="None")
            next_order_type_df = self.__primary(next_parent_order_type)
            for row_id in row_idx:
                keep_idx += next_order_type_df.query(expr=f"dir_{parent_order_type}=={row_id}").index.tolist()
            tmp_df = self.__primary(next_parent_order_type).loc[keep_idx, :]
        return row_idx

    def __get_parent_idx(self, row: dict) -> dict:
        order_type = self.__get_order_types(row, dismiss_secondary=True)[0]
        parent_idx = {order_type: self.df_databases[order_type].query(expr=self.__order_to_query(row)).index[0]}
        infinity_loop_breaker = 0
        while order_type != self.order_type_succession[0]:
            prior_order_type = self.__get_prior_order_type(order_type)
            parent_dir_id = row[f"dir_{prior_order_type}"]
            parent_idx[prior_order_type] = int(parent_dir_id)
            row = self.__primary(prior_order_type).loc[parent_dir_id]
            order_type = self.__get_prior_order_type(order_type)
            if infinity_loop_breaker > 10e3:
                raise TimeoutError(f"Infinity loop apprehended. If your database contains more than "
                                   f"{infinity_loop_breaker} order types raise the breaker value in the code one line "
                                   f"above.")
            infinity_loop_breaker += 1
        return parent_idx

    def __get_parent_order_types(self, order_type: str) -> list[str]:
        return self.order_type_succession[:self.order_type_succession.index(order_type)]

    def __get_next_order_type(self,
                              order_type: str,
                              raise_warning: bool=False,
                              raise_error: bool=False,
                              return_on_overflow=None) -> str or None:
        if order_type == self.order_type_succession[-1]:
            if raise_error:
                raise ValueError(f"Order type {order_type} is the last existing order type, there is no order type "
                                 f"after {order_type}.")
            elif raise_warning:
                helper.warning("Trying to get the order type after the last existing order type. Returning None as "
                               "str or bool instead.")
            return return_on_overflow
        return self.order_type_succession[self.order_type_succession.index(order_type)+1]

    def __get_prior_order_type(self,
                               order_type: str,
                               raise_warning: bool=False,
                               raise_error: bool=False,
                               return_on_overflow=None) -> str or None:
        if order_type == self.order_type_succession[0]:
            if raise_error:
                raise ValueError(f"Order type {order_type} is the first existing order type, there is no order type "
                                 f"before {order_type}.")
            elif raise_warning:
                helper.warning("Trying to get the order type prior to the first existing order type. Returning None as "
                               "str or bool instead.")
            return return_on_overflow
        return self.order_type_succession[self.order_type_succession.index(order_type)-1]

    def __clean_orders(self, order_type: str):
        ignore_some = False if self.__ignore_param_for_order[order_type] is None else True
        if not ignore_some:
            return
        for trigger_param, info_lists in self.__ignore_param_for_order[order_type].items():
            for info_list in info_lists:
                trigger_value = info_list[0]
                ignore_parameters = info_list[1:]
                dict_for_df = helper.merge_list_of_dicts(self.orders[order_type])
                df_all_orders = pd.DataFrame(dict_for_df)
                df_only_trigger = df_all_orders.loc[df_all_orders[trigger_param] == trigger_value]
                if df_only_trigger.empty:
                    helper.warning(f"{trigger_param}={trigger_value} does not exist for this set of parameters. Ignoring "
                                   f"{[trigger_param, [trigger_value]+ignore_parameters]} and continuing.")
                    continue
                df_with_duplicates = df_only_trigger.drop(ignore_parameters, axis=1)
                df_duplicates_cleaned = df_with_duplicates.drop_duplicates()
                df_duplicates_cleaned =  df_duplicates_cleaned.fillna("None")
                df_without_trigger = df_all_orders.loc[df_all_orders[trigger_param] != trigger_value]

                df_all_orders = pd.concat([df_without_trigger, df_duplicates_cleaned], ignore_index=True).fillna("None")
                cleaned_orders = helper.df_to_orders(df_all_orders)
                self.orders[order_type] = cleaned_orders

    def __match_param_order_type(self, param: str) -> str:
        for order_type in self.order_type_succession:
            if param in list((self.primary_parameters[order_type]|self.secondary_parameters[order_type]).keys()):
                return order_type
        raise ValueError(f"Parameter {param} does not exist in any database.")

    def __primary(self, order_type: str) -> pd.DataFrame:
        """Returns a VIEW of the dataframe for a certain order type containing only the primary parameters."""
        return copy(self.df_databases[order_type]).drop(list(self.secondary_parameters[order_type].keys()), axis=1)

    def __secondary(self, order_type: str) -> pd.DataFrame:
        """Returns a VIEW of the dataframe for a certain order type containing only the secondary parameters."""
        return copy(self.df_databases[order_type]).drop(list(self.primary_parameters[order_type].keys()), axis=1)

    def __specific(self, order_type: str):
        """Specific combination of parameters. All params have to be of equal length or of length 1. If the param has
         length 1, then it is used for all combinations. If it has a length grater than one, this function will
         iterate over all of them and combine them with the values of the other parameters of the same index"""
        params = self.primary_parameters[order_type]
        orders = list()
        n_orders = 0
        for value in params.values():
            if len(value) > n_orders:
                n_orders = len(value)
        for order in range(n_orders):
            tmp = dict()
            for key, value in params.items():
                if len(value) > 1:
                    try:
                        tmp[key] = value[order]
                    except IndexError:
                        raise IndexError("All parameters for a certain order type need to have the same number of "
                                         "different parameters or have only one. Parameters of length 1 will always "
                                         "stay the same value, the rest will iterate through the list of parameters.")
                else:
                    tmp[key] = value[0]
            orders.append(tmp)
        return orders

    def __set_params_from_file(self, dataframe: pd.DataFrame, order_type: str):
        primary_params = dict()
        secondary_params = list()
        self.__ignore_param_for_order[order_type] = None
        non_parameter_cols = ["secondaryParameters", "indexExperiment", "ignore"]
        for param in dataframe.columns:
            if param not in non_parameter_cols:
                primary_params[param] = list()
                for value in dataframe[param]:
                    if pd.isna(value):
                        break
                    if type(value) == str:
                        if any(character.isdigit() for character in value):
                            value = ast.literal_eval(value)
                    elif int(value) == value:
                        value = int(value)
                    primary_params[param].append(value)
            elif param == "secondaryParameters":
                secondary_params = dataframe[param].tolist()
            elif param == "ignore":
                ignore_dict = dict()
                for ignore in dataframe[param]:
                    if pd.isna(ignore):
                        break
                    ignore = ast.literal_eval(ignore)
                    ignore_dict[ignore[0]] = ignore[1]
                self.set_ignore_params_for_order(order_type, ignore_dict)
        self.set_params(order_type, **primary_params, secondary_parameters=secondary_params)

    def __full_fac(self, order_type: str):
        """Full factorial parameter combination"""
        params = self.primary_parameters[order_type]
        orders = [{}]
        for param, values in params.items():
            try:
                orders = [comb | {param: value} for comb in orders for value in values]
            except TypeError:
                raise TypeError(f"All parameters have to be given as a list even if they are only one value. "
                                f"Parameter {param} was falsely given as type {type(values)}")
        for id, order in enumerate(copy(orders)):
            orders[id] = self.__dicts_list_list_to_str(order)
        return orders

    def __validate_values_of_order(self, multi_order: dict, order_types: list, dismiss_secondary: bool=False):
        val_order = copy(multi_order)
        for order_type in order_types:
            parameters = [param for param in self.df_databases[order_type] if "dir" not in param]
            if dismiss_secondary:
                parameters = [param for param in parameters if param not in self.secondary_parameters[order_type]]
                for sec_param in self.secondary_parameters[order_type]:
                    try:
                        val_order.pop(sec_param)
                    except KeyError:
                        pass
            for parameter in parameters:
                if multi_order[parameter] not in self.df_databases[order_type][parameter].unique().tolist():
                    raise ValueError(f"{parameter}={multi_order[parameter]} does not exist in "
                                     f"{self.file_databases[order_type]}.")
                val_order.pop(parameter)
        if len(list(val_order.keys())) > 0:
            raise ValueError(f"The parameters {list(val_order.keys())} do not exist in any database.")

    @staticmethod
    def __dicts_list_list_to_str(orders: dict):
        for param, param_value in copy(orders).items():
            if type(param_value) == list:
                orders[param] = str(param_value)  # this is needed so that a df and its read_csv are the same
        return orders

    @staticmethod
    def __order_to_query(order: dict):
        """An order is just a dict that has the parameter names as keys and each value as values. This only works for
        querying dataframes that were created with 'read_csv' (because of the transition from lists and booleans to
        strings)"""
        query = str()
        for param, value in order.items():
            if type(value) in {str, list}:
                query += f"{param}=='{value}' and "
            else:
                query += f"{param}=={value} and "
        return query[:-5]

    @staticmethod
    def __df_to_list_of_dicts(df: pd.DataFrame) -> list[dict]:
        list_of_dicts = list()
        for _, row in df.iterrows():
            row = {key: int(value) for key, value in copy(row).to_dict().items()}
            list_of_dicts.append(row)
        return list_of_dicts


class HamelOseenVortexCreator:
    def __init__(self, working_dir: str, database_file: str):
        self.working_dir = working_dir
        self.df_database = pd.read_csv(database_file)

        self.dir_data = self.working_dir + "/data"
        if not os.path.isdir(self.dir_data):
            helper.create_dir(self.dir_data)
        self.background_truth = None

        self.dir_imgs = None
        self.file_bbox = None
        self.file_additional_img_information = None
        self.arrowhead = None
        self.dpi = None

        self.seed_in = None
        self.width = None
        self.height = None
        self.img_width = None
        self.img_height = None
        self.plot_type = None

        self.bbox_size_x = None
        self.bbox_size_y = None
        self.n_information_per_axis_x = None
        self.n_information_per_axis_y = None
        self.bbox_size_fac = None
        self.crop_rectangle = None

        self.gamma = None
        self.time = None
        self.noise_fac = None
        self.local_noise_fac = None
        self.n_local_noise_areas = None
        self.n_vortices_per_img = None
        self.rand_rot_dir = None

        self.noise_area_covered = "None"
        self.homogeneous_noise = "None"
        self.local_noise = "None"

    def create_pngs(self, dpi: int=10) -> None:
        matplotlib.use("Agg")
        self.dpi = dpi

        for idx_row in range(self.df_database.shape[0]):
            self.__dataframe_row_to_params(self.df_database.iloc[idx_row], idx_row)
            helper.create_dir(self.dir_imgs)
            n_existing_imgs = len(os.listdir(self.dir_imgs))
            if n_existing_imgs == self.n_imgs:
                continue
            elif n_existing_imgs != 0 and n_existing_imgs < self.n_imgs:
                print(f"Unfinished creation of data in {self.dir_imgs}.")
            homogeneous_noise, local_noise, local_noise_area_covered = 0, 0, 0
            df_bbox = pd.DataFrame(columns=["bb_xmin", "bb_ymin", "bb_xmax", "bb_ymax"])
            df_info = pd.DataFrame(columns=["distance", "local_noise_area", "homogeneous_noise", "local_noise"])
            df_bbox.to_csv(self.file_bbox, index=False), df_info.to_csv(self.file_additional_img_information,
                                                                        index=False)
            if self.plot_type == "vec":
                self.crop_rectangle = self.__get_crop_rectangle()

            stretch_gamma, offset_gamma = helper.interval_to_shape_params(self.gamma)
            stretch_time, offset_time = helper.interval_to_shape_params(self.time)

            for idx_img in range(n_existing_imgs, self.n_imgs):
                u, v = self.create_background_flow(self.n_information_per_axis_x,
                                                   self.n_information_per_axis_y,
                                                   self.rn_choice.choice([-1, 1]) * self.rn_u.random(),
                                                   self.rn_choice.choice([-1, 1]) * self.rn_v.random())
                bboxes = list()
                bbox_list_plot = list()
                center_list = [0.4 * self.rn_center.random(2) + 0.3]
                for n_vortices in range(self.rn_choice.choice(self.n_vortices_per_img) - 1):
                    if self.vortex_distance is not None:
                        angle = 2 * np.pi * self.rn_angle_center.random()
                        center_list.append([center_list[0][0] + self.center_distance * np.cos(angle),
                                            center_list[0][1] + self.center_distance * np.sin(angle)])
                    else:
                        center_list.append(0.7 * self.rn_center.random(2) + 0.15)

                rotating_direction = self.rn_choice.choice([-1, 1])
                turn_around = itertools.cycle([1, -1])
                for center in center_list:
                    bboxes.append(self.__get_bbox(center))
                    # bbox_list_plot.append(self.__get_bbox(center, True))  # plot imgs with the bbox

                    if self.rand_rot_dir == "yes":
                        rotating_direction = self.rn_choice.choice([-1, 1])
                    else:
                        rotating_direction *= next(turn_around)
                    gamma = rotating_direction*(stretch_gamma * self.rn_gamma.random() + offset_gamma) # +-[1;2]
                    time = stretch_time * self.rn_time.random() + offset_time  # [80, 90]
                    nu = 14e-6
                    x, y, v_x, v_y = self.__create_vortex_vector_field(center, gamma, nu, time)
                    u += v_x
                    v += v_y
                total_v = np.sqrt(u ** 2 + v ** 2)
                v_max_mean = np.mean([total_v[id] for id in np.argsort(total_v)[::-1][:5]])
                if self.noise_fac != 0:
                    u_noise, v_noise, rms_homogeneous = self.__create_homogeneous_noise()
                    u += u_noise
                    v += v_noise
                    homogeneous_noise = rms_homogeneous / v_max_mean
                if self.local_noise_fac != 0:
                    local_noise_u, local_noise_v, rms_local, local_noise_area_covered = self.__create_local_noise()
                    u += local_noise_u
                    v += local_noise_v
                    local_noise = rms_local / v_max_mean

                u = np.resize(u, (self.n_information_per_axis_x, self.n_information_per_axis_y))
                v = np.resize(v, (self.n_information_per_axis_x, self.n_information_per_axis_y))

                img_info = ({"distance": self.vortex_distance,
                             "homogeneous_noise": homogeneous_noise,
                             "local_noise": local_noise,
                             "local_noise_area": local_noise_area_covered,
                             "n_vortices": len(center_list)})
                df_bbox = df_bbox.append(helper.merge_list_of_dicts(bboxes), ignore_index=True)
                df_info = df_info.append(img_info, ignore_index=True)
                df_bbox.to_csv(self.file_bbox, index=False),
                df_info.to_csv(self.file_additional_img_information, index=False)

                file_img = self.dir_imgs + f"/{idx_img}.png"
                if self.plot_type == "vec":
                    self.__vector_field2image(file_img, x, y, u, v, bbox_list_plot)
                elif self.plot_type == "col":
                    self.vector_field2colour_plot(file_img, u, v)
        return None

    def __create_vortex_vector_field(self,
                                     center: np.array,
                                     gamma: float,
                                     nu: float,
                                     time: float) -> Tuple[np.array, np.array, np.array, np.array]:
        x, y = np.meshgrid(np.linspace(0, 1, self.n_information_per_axis_x),
                           np.linspace(0, 1, self.n_information_per_axis_y))
        samples = np.array([x.ravel(), y.ravel()]).T
        r = np.array([np.linalg.norm(center - s) for s in samples])
        d_x = np.array([center[0] - s[0] for s in samples]).T
        d_y = np.array([center[1] - s[1] for s in samples]).T
        theta = np.arccos(d_x / r) * np.sign(d_y)
        r0 = 2 * np.sqrt(nu * time)
        q = (r / r0)
        v_theta = gamma / (4 * np.pi * np.sqrt(nu * time)) * (1 - np.exp(-q ** 2)) / (q + 1e-16)
        v_x = v_theta * np.sin(-theta)
        v_y = v_theta * np.cos(theta)
        return x, y, v_x, v_y

    def __create_homogeneous_noise(self) -> Tuple[np.array, np.array, float]:
        u_noise = self.noise_fac * self.rn_noise.random(self.n_information_per_axis_x * self.n_information_per_axis_y)
        v_noise = self.noise_fac * self.rn_noise.random(self.n_information_per_axis_x * self.n_information_per_axis_y)
        homogeneous_noise = np.sqrt(u_noise ** 2 + v_noise ** 2)
        rms_homogeneous = rms(homogeneous_noise, self.background_truth)
        return u_noise, v_noise, rms_homogeneous

    def __create_local_noise(self) -> Tuple[np.array, np.array, float, float]:
        local_noise_pos, noise_area_covered = self.__get_local_noise_pos(self.n_local_noise_areas)
        noise_everywhere_u = self.local_noise_fac * (self.rn_local_noise.random([self.n_information_per_axis_x *
                                                                                 self.n_information_per_axis_y]))

        noise_everywhere_v = self.local_noise_fac * (self.rn_local_noise.random([self.n_information_per_axis_x *
                                                                                 self.n_information_per_axis_y]))

        local_noise_u = local_noise_pos * noise_everywhere_u
        local_noise_v = local_noise_pos * noise_everywhere_v

        tmp_local_noise_u, tmp_local_noise_v = copy(local_noise_u), copy(local_noise_v)
        tmp_local_noise_u = tmp_local_noise_u[tmp_local_noise_u != 0]
        tmp_local_noise_v = tmp_local_noise_v[tmp_local_noise_v != 0]
        local_noise = np.sqrt(tmp_local_noise_u ** 2 + tmp_local_noise_v ** 2)
        rms_local = rms(local_noise, self.background_truth[:len(local_noise)])
        return local_noise_u, local_noise_v, rms_local, noise_area_covered

    def __get_bbox(self,
                   center: np.array,
                   plot_bbox=False) -> dict or list:
        if self.plot_type == "col":
            bb_xmin = center[0] * self.n_information_per_axis_x - self.bbox_size_x / 2
            bb_ymin = center[1] * self.n_information_per_axis_y - self.bbox_size_y / 2
            bb_xmin = np.floor(bb_xmin)
            bb_ymin = np.floor(bb_ymin)
            bbox = {
                "bb_xmin": bb_xmin,
                "bb_ymin": self.n_information_per_axis_y - (bb_ymin + self.bbox_size_y),
                "bb_xmax": bb_xmin + self.bbox_size_x,
                "bb_ymax": self.n_information_per_axis_y - bb_ymin,
            }
        elif self.plot_type == "vec":
            bb_xmin = center[0] * self.img_width - self.bbox_size_x / 2
            bb_ymin = center[1] * self.img_height - self.bbox_size_y / 2
            bbox = {
                "bb_xmin": bb_xmin,
                "bb_ymin": self.img_height - (bb_ymin + self.bbox_size_y),
                "bb_xmax": bb_xmin + self.bbox_size_x,
                "bb_ymax": self.img_height - bb_ymin
            }
        if plot_bbox:
            bb_xmin = center[0] - self.bbox_size_fac / 2
            bb_ymin = center[1] - self.bbox_size_fac / 2
            bbox_plot = [bb_xmin, bb_ymin, bb_xmin + self.bbox_size_fac, bb_ymin + self.bbox_size_fac]
            return bbox_plot
        return bbox

    def __get_local_noise_pos(self,
                              n_areas: str) -> Tuple[np.array, float]:
        img = Image.new('1', (self.n_information_per_axis_y, self.n_information_per_axis_x),
                        'black')  # (rows, cols) = (y,x)
        draw = ImageDraw.Draw(img)
        for area in range(self.rn_choice.choice(ast.literal_eval(n_areas))):
            center = (self.rn_local_pos.random() * self.n_information_per_axis_y,
                      self.rn_local_pos.random() * self.n_information_per_axis_x)
            polygon_points = list()
            for point in range(self.rn_choice.randint(3, 20)):
                polygon_points.append(
                    (0.2 * self.n_information_per_axis_y * (2 * self.rn_local_pos.random() - 1) + center[0],
                     0.2 * self.n_information_per_axis_x * (2 * self.rn_local_pos.random() - 1) + center[1]))
            draw.polygon(polygon_points, fill="white")
        noise_area_covered = np.count_nonzero(np.asarray(img)) / (self.n_information_per_axis_y *
                                                                  self.n_information_per_axis_x)
        return np.reshape(np.asarray(img),
                          (self.n_information_per_axis_x * self.n_information_per_axis_y)), noise_area_covered

    def __vector_field2image(self,
                             file_image: str,
                             x: np.array,
                             y: np.array,
                             u: np.array,
                             v: np.array,
                             bbox_list_plot=List[list]) -> None:
        fig, ax = plt.subplots()
        if self.arrowhead == "no":
            ax.quiver(x, y, u, v, headaxislength=0, headlength=0)
        else:
            ax.quiver(x, y, u, v)
        if len(bbox_list_plot) != 0:
            for box in bbox_list_plot:
                plt_box = plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor="orange",
                                        linewidth=10)
                ax.add_patch(plt_box)
        fig.set_size_inches(self.width, self.height)
        plt.axis("off")
        fig.savefig(file_image, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        self.__crop_img(file_image)
        return None

    def __get_crop_rectangle(self) -> Tuple[int, int, int, int]:
        file_temp_img = self.dir_imgs + "temp.png"
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.set_size_inches(self.width, self.height)
        plt.axis("off")
        fig.savefig(file_temp_img, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        img = Image.open(file_temp_img)
        nonwhite_positions = [(x, y) for x in range(img.size[0]) for y in range(img.size[1]) if
                              img.getdata()[x + y * img.size[0]] != (255, 255, 255, 255)]
        rect = (min([x for x, y in nonwhite_positions]), min([y for x, y in nonwhite_positions]),
                max([x for x, y in nonwhite_positions]), max([y for x, y in nonwhite_positions]))
        os.remove(file_temp_img)
        return rect

    def __crop_img(self,
                   file_image: str, ) -> None:
        img = Image.open(file_image)
        img = img.crop(self.crop_rectangle)
        img = img.resize((self.img_width, self.img_height))
        img.save(file_image)
        return None

    def __dataframe_row_to_params(self, df_row_slice: pd.core.series.Series, idx: int):
        row = df_row_slice.to_dict()
        self.dir_imgs = self.dir_data + f"/{idx}/imgs"
        self.file_bbox = self.dir_data + f"/{idx}/bbox.dat"
        self.file_additional_img_information = self.dir_data + f"/{idx}/imgsInformation.dat"

        self.plot_type, self.bbox_size_fac = row["plotType"], row["bboxSizeFac"]
        self.info_axis_type = row["infoAxisType"]
        information_per_axis = ast.literal_eval(row["nInfoPerAxis"])
        if self.plot_type == "vec":
            size, self.arrowhead = ast.literal_eval(row["imgSize"]), row["arrowhead"]
            self.width, self.height = size[0]/self.dpi, size[1]/self.dpi
            self.img_width, self.img_height = int(self.width*self.dpi), int(self.height*self.dpi)
            self.bbox_size_x = self.bbox_size_fac * self.img_width
            self.bbox_size_y = self.bbox_size_fac * self.img_height

            if self.info_axis_type == "percentage":
                self.n_information_per_axis_x = int(self.img_width*float(row["percentageInfoAxis"])/100)
                self.n_information_per_axis_y = int(self.img_height*float(row["percentageInfoAxis"])/100)
            else:
                self.n_information_per_axis_x = information_per_axis[0]
                self.n_information_per_axis_y = information_per_axis[1]
        else:
            self.n_information_per_axis_x = information_per_axis[0]
            self.n_information_per_axis_y = information_per_axis[1]
            self.bbox_size_x = self.bbox_size_fac * self.n_information_per_axis_x
            self.bbox_size_y = self.bbox_size_fac * self.n_information_per_axis_y


        self.gamma, self.time, self.n_imgs = ast.literal_eval(row["gamma"]), ast.literal_eval(row["time"]), row["nImg"]
        self.noise_fac, self.local_noise_fac = row["noiseFac"], row["localNoiseFac"]
        self.n_local_noise_areas = row["nLocalNoiseAreas"]
        self.n_vortices_per_img = ast.literal_eval(row["nVortices"])

        self.vortex_distance = row["presetDistance"]
        self.rand_rot_dir = row["randRotDir"]
        self.center_distance = self.bbox_size_fac * self.vortex_distance
        self.background_truth = np.zeros((self.n_information_per_axis_x * self.n_information_per_axis_y))
        self.__set_seeds(row["seed"])

    def __set_seeds(self, seed_in):
        generator_seed = np.random.RandomState(seed_in)
        self.rn_center = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_gamma = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_nu = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_time = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_choice = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_u = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_v = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_noise = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_local_noise = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_local_pos = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))
        self.rn_angle_center = np.random.RandomState(int(np.around(generator_seed.random() * 1000)))

    @staticmethod
    def create_background_flow(n_information_per_axis_x: int,
                               n_information_per_axis_y: int,
                               u=0,
                               v=0) -> Tuple[np.array, np.array]:
        grid = np.ones(n_information_per_axis_x * n_information_per_axis_y)
        return u * grid, v * grid

    def vector_field2colour_plot(self,
                                 file_image: str,
                                 u: np.array,
                                 v: np.array) -> None:
        n_x, n_y = u.shape
        u = np.resize(u, (n_x * n_y, 1))
        v = np.resize(v, (n_x * n_y, 1))

        u, v = self.__scale_velocity_field(u, v)

        u = np.resize(u, (n_x, n_y))
        v = np.resize(v, (n_x, n_y))

        u = np.flipud(u)
        v = np.flipud(v)
        b = np.ones((n_x, n_y))
        alpha = np.ones((n_x, n_y))

        rgba_unit8 = (np.dstack((u, v, b, alpha)) * 255.999).astype(np.uint8)
        image = Image.fromarray(rgba_unit8)
        image.save(file_image)
        return None

    @staticmethod
    def __scale_velocity_field(*args, scaler_class=MinMaxScaler):
        """Scaler: CLASS (not instance!) of a scaler"""
        scaler = scaler_class()
        scaled = list()
        for arg in args:
            scaler.fit(arg)
            scaled.append(scaler.transform(arg))
        return tuple(scaled)

    @staticmethod
    def add_bboxes(working_dir: str,
                   colour: str,
                   train_or_test="train" or "test"):
        df_bbox = pd.read_csv(working_dir + "bbox_" + train_or_test + ".dat", index_col=None)
        n_imgs, _ = df_bbox.shape
        original_dir = working_dir + train_or_test + "/"
        bbox_dir = working_dir + train_or_test + "_bbox/"
        hf.create_directory(bbox_dir)
        for index in range(n_imgs):
            file_image = original_dir + f"{index}.png"
            boxes, _ = helper.pd_row2(dataframe=df_bbox, index_col=index, to="list")
            im = Image.open(file_image)
            draw = ImageDraw.Draw(im)
            for box in boxes:
                draw.rectangle(box, fill=None, outline=colour, width=2)
            im.save(bbox_dir + f"{index}.png")


class NonAnalyticalData:
    def __init__(self):
        pass

    def plot_mat(
            self,
            file: str,
            percentage_information_x_axis: float,
            percentage_information_y_axis: float,
            plot_as: str = "vec" or "col",
            normalise: bool = False,
            mean: bool = False
    ) -> None:
        vec = True if plot_as == "vec" else False
        if mean and vec:
            print("Colourised velocity fields are always averaged.")
        if percentage_information_x_axis > 100 or percentage_information_y_axis > 100:
            raise ValueError("There is no more than 100% information.")

        root = "D:/Jonas/Studium/TU/Semester/Bachelor/pythonProject/exercise_object_detection/data_experimental"
        mat = io.loadmat(file)
        dir_save = root + "/" + file[file.rfind("/"):file.find(".")] + f"_{plot_as}"
        if mean and vec:
            dir_save += "_mean"
        if normalise:
            dir_save += "_normalise"
        hf.create_directory(dir_save)

        rows = mat["U"].shape[0]
        upper_lim = rows - 1
        upper_lim_found = False
        while not upper_lim_found:
            if mat["U"][upper_lim, 0, 0] != 0:
                upper_lim_found = True
            else:
                upper_lim -= 1
        lower_lim = 0
        lower_lim_found = False
        while not lower_lim_found:
            if mat["U"][lower_lim, 0, 0] != 0:
                lower_lim_found = True
            lower_lim += 1

        n_used_cols = mat["U"].shape[1]
        n_used_rows = upper_lim - lower_lim
        n_wanted_cols = n_used_cols * percentage_information_x_axis / 100
        n_wanted_rows = n_used_rows * percentage_information_y_axis / 100
        step_x = n_used_cols / n_wanted_cols
        step_y = n_used_rows / n_wanted_rows
        cols2use = [int(np.round(step_x * i)) for i in range(int(np.floor(n_wanted_cols)))]
        rows2use = [int(np.round(step_y * i)) for i in range(int(np.floor(n_wanted_rows)))]

        no_measurements = mat["t"].shape[1]
        x = np.asarray(mat["X"][lower_lim:upper_lim, :])
        y = np.asarray(mat["Y"][lower_lim:upper_lim, :])
        if vec:
            x = x[:, cols2use]
            x = x[rows2use, :]
            y = y[:, cols2use]
            y = y[rows2use, :]
        for i in range(no_measurements):
            mat_u = np.asarray(mat["U"][lower_lim:upper_lim, :, i])
            mat_v = np.asarray(mat["V"][lower_lim:upper_lim, :, i])
            file = dir_save + f"/{i}.png"
            if vec:
                mat_u = mat_u[:, cols2use]
                mat_u = mat_u[rows2use, :]
                mat_v = mat_v[:, cols2use]
                mat_v = mat_v[rows2use, :]
                if mean:
                    mat_u -= mat_u.mean()
                    mat_v -= mat_v.mean()
                if normalise:
                    facs2normalise = np.sqrt(mat_u ** 2 + mat_v ** 2)
                    mat_u = mat_u / facs2normalise
                    mat_v = mat_v / facs2normalise
                self.vector_field2image(file, x, y, mat_u, mat_v, list(list()))
            else:
                if normalise:
                    facs2normalise = np.sqrt(mat_u ** 2 + mat_v ** 2)
                    mat_u = mat_u / facs2normalise
                    mat_v = mat_u / facs2normalise
                file = dir_save + f"/{i}.png"
                self.vector_field2colour_plot(file, mat_u, mat_v)

    @staticmethod
    def vector_field2image(file_image: str,
                           x: np.array,
                           y: np.array,
                           u: np.array,
                           v: np.array,
                           bbox_list_plot=List[list]) -> None:
        fig, ax = plt.subplots()
        if not True:
            ax.quiver(x, y, u, v, headaxislength=0, headlength=0)
        else:
            ax.quiver(x, y, u, v)
        if len(bbox_list_plot) != 0:
            for box in bbox_list_plot:
                plt_box = plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor="orange",
                                        linewidth=10)
                ax.add_patch(plt_box)
        fig.set_size_inches(1590, 990)
        plt.axis("off")
        fig.savefig(file_image, dpi=1, bbox_inches="tight")
        plt.close(fig)
        return None
