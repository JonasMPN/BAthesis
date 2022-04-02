from vortex_detection import data_prep as prep
from helper_functions import Helper
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
        except FileNotFoundError:
            self.df_protocol = pd.DataFrame()

    def evaluate_predictions(self, ignore_params: list[str]):
        evaluation = {"mean_ap": list(), "mean_distance": list(), "no_detections": list()}
        tmp_ids = self.orders.get_all_parents_dir_idx("test")[0]
        file_additional_info = self.dir_root + f"/data/{tmp_ids['data']}/imgsInformation.dat"
        additional_params = pd.read_csv(file_additional_info).columns.tolist()
        additional_params = [param for param in additional_params if param != "vortex_distance"]  # todo hardcoded
        for param in self.orders.get_all_params() + additional_params:
            if param not in ignore_params:
                evaluation[param] = list()

        n_evaluation_orders = self.orders.get_number_of_orders("test")
        for idx_evaluate, dir_ids in enumerate(self.orders.get_all_parents_dir_idx("test")):
            helper.print_progress(idx_evaluate, n_evaluation_orders, "Evaluating")

            file_additional_info = self.dir_root + f"/data/{dir_ids['data']}/imgsInformation.dat"
            additional_info = self.__mean_additional_information(pd.read_csv(file_additional_info, index_col=None))
            evaluation_values = self.__evaluate_dir_ids(dir_ids)
            param_values = self.orders.get_full_params_from_dir_ids(dir_ids)
            for param in copy(param_values).keys():
                if param in ignore_params:
                    param_values.pop(param)

            for param in copy(param_values).keys():
                if "dir" in param:
                    param_values.pop(param)
            for param, value in (evaluation_values | additional_info | param_values).items():
                evaluation[param].append(value)
        pd.DataFrame(evaluation).to_csv(self.file_protocol, index=False)

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
        :param criteria: {"param": "bigger" or "smaller"}
        :param ignore_cols:
        :param mean_cols:
        :return:
        """
        criteria_param = next(iter(criteria))
        better = criteria[criteria_param]
        if better not in ["bigger", "smaller"]:
            raise ValueError(f"Parameter 'criteria' must be a dict with")
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
        df_param_combinations = df_protocol.drop([compare_col, criteria_param] + mean_cols + result_cols, axis=1)
        compare_overview = list()
        n_param_combinations = 0
        while df_param_combinations.shape[0] > 0:
            ids = self.__filter_param_combination(df_param_combinations).index.tolist()
            df_filtered = df_protocol.loc[ids]
            best_idx = df_filtered[criteria_param].idxmax() if better == "bigger" else \
                       df_filtered[criteria_param].idxmin()
            compare_overview.append(df_protocol[compare_col].loc[best_idx])
            df_param_combinations = df_param_combinations.drop(ids)
            n_param_combinations += 1

        compare_values = df_protocol[compare_col].unique().tolist()
        mean_values = {mean_param: {compare_value: None for compare_value in compare_values} for mean_param in
                       [f"{criteria_param}_best"]+mean_cols}
        for compare_value in compare_values:
            df_filtered = df_protocol.loc[df_protocol[compare_col] == compare_value]
            for mean_col in mean_cols:
                mean_values[mean_col][compare_value] = df_filtered[mean_col].mean()
            mean_values[f"{criteria_param}_best"][compare_value] = compare_overview.count(
                compare_value)/n_param_combinations
            #  The above line is not actually a mean value.
        df_mean_values = pd.DataFrame(mean_values)
        for col in mean_cols:
            df_mean_values[col] = df_mean_values[col]/df_mean_values[col].max()
        df_mean_values = df_mean_values.T if plot_as == "bar" else df_mean_values
        plotter[plot_as](save_directory, compare_col, n_param_combinations, df_mean_values, 16, 9, 100)

    def __evaluate_dir_ids(self, dir_ids: dict) -> dict:
        current_data_dir = self.dir_root + f"/data/{dir_ids['data']}"
        file_bbox = current_data_dir + "/bbox.dat"
        dir_train = current_data_dir + f"/{dir_ids['train']}"
        dir_test = dir_train + f"/{dir_ids['test']}"
        prediction_file = dir_test + "/predictions.dat"

        self.set_truth(file_bbox)
        self.set_predictions(prediction_file)
        mean_ap, mean_distance = self.get_all_criteria_mean()
        return {"mean_ap": mean_ap, "mean_distance": mean_distance[0], "no_detections": mean_distance[1]}

    def __mean_additional_information(self, df_additional_information: pd.DataFrame) -> dict:
        averaged = dict()
        for param in df_additional_information.columns.tolist():
            averaged[param] = df_additional_information[param].mean()
        return averaged

    @staticmethod
    def __bar_plots(save_directory: str,
                    compare_param: str,
                    n_param_combinations: int,
                    dataframe: pd.DataFrame,
                    img_width: float,
                    img_height: float,
                    dpi: int):
        fig, ax = plt.subplots()
        fig.set_size_inches(img_width, img_height)
        dataframe.plot.bar(ax=ax)
        plt.title(f"Compared parameter: {compare_param}. Parameter combinations: {n_param_combinations}")
        plt.xticks(rotation="horizontal")
        plt.savefig(save_directory+f"/compare_{compare_param}_bar.png", dpi=dpi)
        plt.close(fig)

    @staticmethod
    def __pie_plots(save_directory: str,
                    compare_param: str,
                    n_param_combinations: int,
                    dataframe: pd.DataFrame,
                    img_width: float,
                    img_height: float,
                    dpi: int):
        for col in dataframe.columns:
            fig, ax = plt.subplots()
            fig.set_size_inches(img_width, img_height)
            dataframe[col].plot.pie(y=col, ax=ax)
            plt.title(f"Compared parameter: {compare_param}. Parameter combinations: {n_param_combinations}")
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



