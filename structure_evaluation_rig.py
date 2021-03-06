import ast, os

from vortex_detection import data_prep as prep
from helper_functions import Helper
from numpy import round
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
from vortex_detection.data_evaluation import PredictionEvaluator

helper = Helper()


class EvaluationRig(PredictionEvaluator):
    def __init__(self,
                 orders: prep.Orders,
                 root_dir: str,
                 protocol_file: str):
        PredictionEvaluator.__init__(self)
        self.orders = orders
        self.dir_root = root_dir
        self.file_protocol = protocol_file
        try:
            self.df_protocol = pd.read_csv(self.file_protocol, index_col=None)
            self.already_evaluated = int(self.df_protocol["dir_test"].max())
        except FileNotFoundError:
            self.df_protocol = pd.DataFrame()
            self.already_evaluated = -1

    def predictions(self, handle_additional_information: dict, ignore_params: list[str],
                    validation_cols: list[str]=None, all: bool=True):
        """

        :param ignore_params:
        :param handle_additional_information: set how to handle each parameter of the additional information file.
        Handle schemes are: 'mean' and 'sum'. Use each parameter individually as a key or use 'all' to set all at
        once. If 'all' is used, keys of individual parameters will overwrite the scheme of 'all'.
        :param all: if True then the evaluation does not depend on the current set of orders, else only the orders
        will be evaluated.
        :return:
        """
        evaluation = {"dir_test": list(), "mean_ap": list(), "mean_distance_normalised": list(),
                      "detected_normalised":list()}
        tmp_ids = self.orders.get_all_parents_dir_idx("test")[0]
        file_additional_info = self.dir_root + f"/data/{tmp_ids['data']}/imgsInformation.dat"
        additional_params = pd.read_csv(file_additional_info).columns.tolist()
        additional_params = [param for param in additional_params if param != "vortexDistance"]  # todo hardcoded
        handle_dict = dict()
        if "all" in handle_additional_information.keys():
            scheme = handle_additional_information["all"]
            for param in additional_params:
                handle_dict[param] = scheme
            handle_additional_information.pop("all")
        for param, handle in handle_additional_information.items():
            handle_dict[param] = handle

        for param in self.orders.get_all_params() + additional_params:
            if param not in ignore_params:
                evaluation[param] = list()

        if all:
            df_test = pd.read_csv(self.dir_root+f"/database_test.dat", index_col=None)
            dir_indices = list()
            start_id = self.already_evaluated if self.already_evaluated != -1 else 0
            for dir_test_id in range(start_id, int(df_test.shape[0])):
                dir_indices.append(self.orders.get_parent_dir_idx({"test": dir_test_id}))
            n_evaluation_orders = len(dir_indices)
        else:
            dir_indices = self.orders.get_all_parents_dir_idx("test")
            n_evaluation_orders = len(dir_indices)

        for idx_evaluate, dir_ids in enumerate(dir_indices):
            if dir_ids["test"] <= self.already_evaluated:
                print(f"Already evaluated test directory {dir_ids['test']}.")
                continue
            helper.print_progress(idx_evaluate, n_evaluation_orders, "Evaluating")
            evaluation["dir_test"].append(dir_ids["test"])

            file_additional_info = self.dir_root + f"/data/{dir_ids['data']}/imgsInformation.dat"
            df_additional_info = pd.read_csv(file_additional_info, index_col=None)
            test_interval = self.__get_test_img_interval(dir_ids)
            test_range = range(int(test_interval[0]), int(test_interval[1]))
            additional_info = self.__handle_additional_information(df_additional_info.loc[test_range], handle_dict)
            evaluation_values = self.__evaluate_dir_ids(dir_ids, additional_info["n_vortices"])
            param_values = self.orders.get_full_params_from_dir_ids(dir_ids)
            for param in copy(param_values).keys():
                if param in ignore_params or "dir" in param:
                    param_values.pop(param)

            for param, value in (evaluation_values | additional_info | param_values).items():
                evaluation[param].append(value)

        df_new_eval = pd.DataFrame(evaluation)
        if validation_cols is not None:
            df_new_eval, _ = self.__parse_validation_columns(dataframe=df_new_eval, columns=validation_cols)
        self.df_protocol = pd.concat([self.df_protocol, df_new_eval], ignore_index=True)
        self.df_protocol["dir_test"].astype("int")
        self.df_protocol.to_csv(self.file_protocol, index=False)

    def compare(self,
                save_directory: str,
                compare_col: str,
                criteria: dict,
                result_cols: list[str],
                mean_cols: list[str]=None,
                ignore_cols: list[str]=None,
                plot_as: str="bar"):
        """
        :param save_directory:
        :param param:
        :param result_cols: columns that represent parameters that have not been set, but are results of an experiment
        :param criteria: {"param": "bigger" or "smaller", "backup_param": "bigger" or "smaller"}
        :param ignore_cols: columns that will be ignored
        :param mean_cols: columns
        :return:
        """
        if not os.path.isdir(save_directory):
            helper.create_dir(save_directory)

        criteria_keys = iter(criteria)
        criteria_param, backup_param = next(criteria_keys), next(criteria_keys)
        better = criteria[criteria_param]
        better_backup = criteria[backup_param]
        if better not in ["bigger", "smaller"]:
            raise ValueError(f"Parameter 'criteria' must be a dict with a value of either 'bigger' or 'smaller'.")
        for ignore in ignore_cols:
            if ignore in result_cols+mean_cols:
                raise ValueError(f"Ignored column {ignore} must not be existing in result_cols or mean_cols.")
        if plot_as not in ["bar", "pie"]:
            raise ValueError(f"Parameter plot_as must be either 'bar' or 'pie'. Plotting as {plot_as} is not "
                             f"supported.")
        ignore_cols = ignore_cols if ignore_cols is not None else list()
        mean_cols = mean_cols if mean_cols is not None else list()
        df_protocol = copy(self.df_protocol)
        existing_params = df_protocol.columns.tolist()
        for parameter in [compare_col, criteria_param] + ignore_cols + mean_cols:
            if parameter not in existing_params:
                raise ValueError(f"Parameter {parameter} does not exist in the file {self.file_protocol}.")
        plotter = {"bar": self.__bar_plots, "pie": self.__pie_plots}

        df_protocol = df_protocol.drop(ignore_cols, axis=1)
        drop_cols = ["dir_test", compare_col, criteria_param]+mean_cols+result_cols
        df_param_combinations = df_protocol.drop(drop_cols, axis=1)
        compare_overview = list()
        n_param_combinations = 0
        counter = 0
        while df_param_combinations.shape[0] > 0:
            # ids = self.__filter_param_combination(df_param_combinations).index.tolist()
            ids = [16*i+counter for i in range(2)]
            df_filtered = df_protocol.loc[ids]
            best_idx = None
            for param, better_str in criteria.items():
                better = max if better_str == "bigger" else min
                best_value = better(df_filtered[param].dropna())
                df_filtered = df_filtered.loc[df_filtered[param] == best_value]
                if df_filtered.shape[0] == 1:
                    best_idx = df_filtered.index[0]
                    break
            if best_idx is None:
                helper.warning(f"During the comparison of {self.file_protocol} there were multiple values for the "
                               f"comparison parameter that had the exact same criteria values. Add more criteria "
                               f"criteria parameters to identify the single best comparison value. \n"
                               f"The smallest row index will now be chosen.")
                best_idx = df_filtered.index[0]

            compare_overview.append(df_protocol[compare_col].loc[best_idx])
            df_param_combinations = df_param_combinations.drop(ids)
            n_param_combinations += 1
            counter += 1

        compare_values = df_protocol[compare_col].unique().tolist()
        mean_values = {mean_param: {compare_value: None for compare_value in compare_values} for mean_param in
                       ["criteria_best"]+mean_cols}
        for compare_value in compare_values:
            df_filtered = df_protocol.loc[df_protocol[compare_col] == compare_value]
            for mean_col in mean_cols:
                mean_values[mean_col][compare_value] = df_filtered[mean_col].mean()
            mean_values["criteria_best"][compare_value] = compare_overview.count(
                compare_value)/n_param_combinations
            #  The above line is not actually a mean value.
        df_mean_values = pd.DataFrame(mean_values)
        max_values = dict()
        for col in mean_cols:
            max_value = df_mean_values[col].max()
            df_mean_values[col] = df_mean_values[col]/max_value
            max_values[col] = round(max_value, 3)
        df_mean_values = df_mean_values.T if plot_as == "bar" else df_mean_values
        title = f"Compared parameter: {compare_col}. Parameter combinations: {n_param_combinations}. \n" \
                f"Best criteria: {criteria}."
        plotter[plot_as](save_directory, compare_col, title, df_mean_values, 16, 9, 100, max_values=max_values)

    def __evaluate_dir_ids(self, dir_ids: dict, n_vortices: int) -> dict:
        current_data_dir = self.dir_root + f"/data/{dir_ids['data']}"
        file_bbox = current_data_dir + "/bbox.dat"
        dir_train = current_data_dir + f"/{dir_ids['train']}"
        dir_test = dir_train + f"/{dir_ids['test']}"
        prediction_file = dir_test + "/predictions.dat"
        if not os.path.isfile(prediction_file):
            return {"mean_ap": None, "mean_distance_normalised": None, "detected_normalised": None}

        self.set_truth(file_bbox)
        self.set_predictions(prediction_file)
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
        results, n_detected = self.get_all_criteria()
        if self.orders.get_value_for(dir_ids, "plotType") == "vec":
            width = helper.str_list2value(self.orders.get_value_for(dir_ids, "imgSize"),0)
        elif self.orders.get_value_for(dir_ids, "plotType") == "col":
            width = helper.str_list2value(self.orders.get_value_for(dir_ids, "nInfoPerAxis"), 0)
        else:
            raise NotImplementedError(f"plotType cannot be something else than vec or col.")
        mean_distance_normalised = results["distance"]/width
        detected_normalised = n_detected/n_vortices
        return {"mean_ap": results["ap"], "mean_distance_normalised": mean_distance_normalised,
                "detected_normalised": detected_normalised}

    def __handle_additional_information(self, df_additional_information: pd.DataFrame, handle_dict: dict) -> dict:
        handled = dict()
        for param in df_additional_information.columns.tolist():
            if handle_dict[param] == "mean":
                handled[param] = df_additional_information[param].mean()
            elif handle_dict[param] == "sum":
                handled[param] = df_additional_information[param].sum()
        return handled

    def __get_test_img_interval(self, dir_ids):
        n_train_imgs = self.orders.get_value_for(dir_ids, "nImgTrain")
        validation = self.orders.get_value_for(dir_ids, "validationInfo")
        n_validation_imgs = helper.str_list2value(validation, 0) * helper.str_list2value(validation, 1)
        test_min = n_train_imgs + n_validation_imgs
        test_max = test_min + self.orders.get_value_for(dir_ids, "nImgTest")
        return [test_min, test_max]

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

    @staticmethod
    def __parse_validation_columns(dataframe: pd.DataFrame, columns: list[str]):
        """
        Columns have to be of shape [[idx_dataset_0, [parameter]], [idx_dataset_1, [parameter]], ...]. This function
        counts the occurrences of each parameter for every row.
        :param columns:
        :return:
        """
        dataframe = dataframe.astype({col: "str" for col in columns})
        new_columns = {col: dict() for col in columns}
        column_lengths = dataframe.shape[0]
        for column in columns:
            row = dataframe[column]
            for row_id, expression in enumerate(row):
                expression = ast.literal_eval(expression)
                new_row_entries = list()
                for entry in expression:
                    for value in entry[1]:
                        new_row_entries.append(value)
                unique_values = set(new_row_entries)
                for unique in unique_values:
                    if unique not in new_columns[column].keys():
                        new_columns[column][unique] = [0 for _ in range(row_id)]
                    new_columns[column][unique].append(new_row_entries.count(unique))
                for missing in [param for param in new_columns[column].keys() if param not in unique_values]:
                    new_columns[column][missing].append(0)
            for param, values in new_columns[column].items():
                if len(values) != column_lengths:
                    new_columns[column][param] += [0 for _ in range(column_lengths - len(values))]
        new_df_protocol = dataframe.drop(columns, axis=1)
        added_columns = list()
        for column in columns:
            add_columns = new_columns[column]
            added_columns += list(add_columns.keys())
            for column_name, values in add_columns.items():
                new_df_protocol[column_name] = values
        return new_df_protocol, added_columns

    @staticmethod
    def __bar_plots(save_directory: str,
                    compare_param: str,
                    title: str,
                    dataframe: pd.DataFrame,
                    img_width: float,
                    img_height: float,
                    dpi: int,
                    **kwargs):
        fig, ax = plt.subplots()
        fig.set_size_inches(img_width, img_height)
        dataframe.plot.bar(ax=ax)
        plt.title(title)
        plt.xticks(rotation="horizontal")
        new_labels = list()
        for Text in ax.get_xticklabels():
            label = Text.get_text()
            if "best" in label:
                new_labels.append(label)
                continue
            new_labels.append(f"{label} \n max: {kwargs['max_values'][label]}")
        ax.set_xticklabels(new_labels)
        plt.savefig(save_directory+f"/compare_{compare_param}_bar.png", dpi=dpi)
        plt.close(fig)

    @staticmethod
    def __pie_plots(save_directory: str,
                    compare_param: str,
                    title: str,
                    dataframe: pd.DataFrame,
                    img_width: float,
                    img_height: float,
                    dpi: int,
                    **kwargs):
        for col in dataframe.columns:
            fig, ax = plt.subplots()
            fig.set_size_inches(img_width, img_height)
            dataframe[col].plot.pie(y=col, ax=ax)
            plt.title(title)
            plt.savefig(save_directory + f"/compare_{compare_param}_pie_{col}.png", dpi=dpi)
            plt.close(fig)

    @staticmethod
    def __filter_param_combination(dataframe: pd.DataFrame) -> pd.DataFrame:
        df_iter = copy(dataframe)
        n_rows = df_iter.shape[0]
        for param in df_iter.columns.tolist():
            if len(dataframe[param].unique()) == 1:
                continue
            dataframe = dataframe.loc[dataframe[param]==dataframe[param].iloc[0]]
            if n_rows == dataframe.shape[0]:
                break
            n_rows = dataframe.shape[0]
        return dataframe





