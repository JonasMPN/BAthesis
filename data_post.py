from helper_functions import Helper
from typing import List
from matplotlib import pyplot as plt
import pandas.plotting as pplot
import matplotlib.axes
import ast
import pandas as pd
import pandas.errors
import numpy as np
from copy import copy
helper = Helper()

class PostVisualisation():
    def __init__(self, dir_results: str, protocol_file: str, criteria_column: str, single_params_file: str,
                 columns_to_ignore: List[str]=None):
        cols_to_ignore = [] if columns_to_ignore is None else columns_to_ignore
        self.dir_results = dir_results
        self.criteria_col = criteria_column
        helper.create_dir(self.dir_results)
        try:
            self.df = pd.read_csv(protocol_file, index_col=None).drop(columns=cols_to_ignore)
            self.df = self.df.loc[self.df["nVortices"] == "[1]"]
            if "None" in self.df[self.criteria_col].unique():
                max_distance = self.df[self.criteria_col].loc[self.df[self.criteria_col] != "None"].max()
                self.df[self.criteria_col] = self.df[self.criteria_col].replace({"None": max_distance})
                self.df[self.criteria_col] = pd.to_numeric(self.df[self.criteria_col])
        except (FileNotFoundError, pandas.errors.EmptyDataError) as e:
            raise e
        self.criteria_min, self.criteria_max = None, None
        self.df_original = pd.read_csv(protocol_file, index_col=None)
        self.unchanged_params = list()
        self.file_single_params = single_params_file
        try:
            self.df_single_params = pd.read_csv(single_params_file, index_col=None)
        except (FileNotFoundError, pandas.errors.EmptyDataError):
            self.df_single_params = pd.DataFrame(columns=["combination", "parameter"])
            self.df_single_params.to_csv(single_params_file, index=False)
        if self.df_single_params is None:
            raise LookupError("Could not create or use an existing file for the single parameters.")
        self._background_category = "_background"  # colour lines w/o category in parallel coordinates

    def parallel_coordinates(self, categories: dict, show_avg_categories=True):
        self.__clean_df()
        self.__translate_df()
        self.__categorise(categories)
        self.__normalise_df(["species"])
        plot_dfs = self.__get_plot_df((categories|{self._background_category:[0,0,"grey"]}), 0.2 if show_avg_categories
                                       else 1)
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)

        for df_info in plot_dfs:
            df, colour, alpha = df_info["df"], df_info["colour"], df_info["alpha"]
            pplot.parallel_coordinates(ax=ax, frame=df, class_column="species", color=colour, alpha=alpha)
        if show_avg_categories:
            for df_info in plot_dfs[1:]: # the first one is the background
                df, colour, alpha = df_info["df"], df_info["colour"], df_info["alpha"]
                df_averaged = pd.DataFrame()
                for col in df.columns.tolist():
                    if col == "species":
                        df_averaged["species"] = f"_{df[col].iloc[0]}"
                    else:
                        df_averaged[col] = [df[col].mean()]
                pplot.parallel_coordinates(ax=ax, frame=df_averaged, class_column="species", color=colour, linewidth=3)
        plt.title(f"{self.criteria_col}=[{np.round(self.criteria_min, 3)}, {np.round(self.criteria_max, 3)}]")
        plt.xticks(rotation="vertical")
        plt.savefig(self.dir_results+"/parallel_coordinates.png", dpi=180, bbox_inches="tight")
        plt.cla()
        plt.clf()

    def single_parameter(self, parameter: str, criteria_col: str, non_parameter_cols: list[str]=None, amount: int=None):
        idx_species, param_values = self.__get_single_param_combinations(parameter, criteria_col, non_parameter_cols)
        for combination, idx in idx_species.items():
            self.df_single_params = self.df_single_params.append({"combination":combination, "parameter": parameter},
                                                                 ignore_index=True)
        self.df_single_params.to_csv(self.file_single_params, index=None)
        self.__plot_single_param(parameter, criteria_col, amount)

    def single_parameter_unique_values(self, parameter: str, criteria_col: str, non_parameter_cols: list[str]=None,
                                       amount: int=None):
        plot_combinations = self.__get_single_param_unique_combinations(parameter, criteria_col, non_parameter_cols)
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        counter = 0
        for _, values in plot_combinations.items():
            if counter == amount:
                break
            ax.plot(values[parameter], values[criteria_col])
            counter += 1
        plt.title(f"Parameter: {parameter}"), plt.xlabel(parameter), plt.ylabel(criteria_col), plt.grid()
        # ax.set_xticks(self.df_original[parameter].unique())
        plt.savefig(self.dir_results+f"/{parameter}.png", dpi=180, bbox_inches="tight")
        plt.cla(), plt.clf()

    def __get_plot_df(self, categories: dict, non_background_alpha: float) -> List[dict]:
        df_single_species = list()
        species = [self._background_category] + [col for col in self.df["species"].unique() if col != self._background_category]
        for species in species:
            df_single_species.append({"df": self.df.loc[self.df["species"] == species],
                                      "colour": categories[species][2],
                                      "alpha": non_background_alpha if species != self._background_category else 0.05})
        return df_single_species

    def __clean_df(self):
        """
        Drops parameters (columns) that have a single fixed value throughout the protocol.
        :return:
        """
        for col in copy(self.df.columns).tolist():
            if len(self.df[col].unique()) == 1 or "dir" in col:
                self.df = self.df.drop([col], axis=1)
                self.unchanged_params.append(col)

    def __translate_df(self):
        """
        Translates parameter values that are not floats into ascending numbers. These numbers will be saved to an
        additional file in the result_dir.
        :return:
        """
        dtypes = dict(self.df.dtypes)
        translation = list()
        for col, dtype in dtypes.items():
            if dtype == "object":
                translate = dict()
                unique_values = self.df[col].unique()
                for id, value in enumerate(unique_values):
                    translate[value] = id
                translation.append({col: translate})
        if len(translation) != 0:
            for to_replace in translation:
                self.df = self.df.replace(to_replace)
            transl_for_df = dict()
            for transl in translation:
                param = next(iter(transl))
                transl_for_df[param] = [param_value for param_value in transl[param].keys()]
            df_translation = pd.DataFrame.from_dict(transl_for_df, orient="index").transpose()
            pd.DataFrame(df_translation).to_csv(self.dir_results+"/translation.dat", index=False)

    def __categorise(self, categories: dict):
        """
        This function categorises the criteria column into the categories of 'categories'.
        :param categories:
        :return:
        """
        self.criteria_min, self.criteria_max = self.df[self.criteria_col].min(), self.df[self.criteria_col].max()
        categories = self.__fill_categories(categories)
        species_idx = {category: list() for category in categories.values()}
        col_to_categorise = copy(self.df[self.criteria_col])
        n_values = len(col_to_categorise)
        last_percentage, used_idx = 0, list()
        for next_percentage, category in categories.items():
            biggest_n = int(np.round(n_values/100*(next_percentage-last_percentage)))
            species_id = col_to_categorise.nlargest(biggest_n).index.tolist()
            species_idx[category] += species_id
            col_to_categorise = col_to_categorise.drop(species_id)
            last_percentage = next_percentage
            used_idx += species_id

        if len(used_idx) != self.df[self.criteria_col].shape:
            species_idx[list(categories.values())[-1]] += col_to_categorise.index.tolist()
        self.df["species"] = None

        for category, idx in species_idx.items():
            for id in idx:
                self.df.at[id, "species"] = category
        self.df = self.df.sort_values("species", ascending=False)

    def __normalise_df(self, cols_to_ignore: list):
        """
        Normalises the columns of a dataframe.
        :param cols_to_ignore:
        :return:
        """
        cols_to_normalise = [col for col in self.df.columns.tolist() if col not in cols_to_ignore]
        for col in cols_to_normalise:
            self.df[col] = self.df[col]/self.df[col].max()

    def __plot_avg_categories(self, ax: matplotlib.axes.Axes, df: pd.DataFrame, categories: dict[str:list]):
        cols_to_mean = [col for col in df.columns.tolist() if df[col].dtype != "object"]
        for category, category_info in categories.items():
            df_categorised = df.loc[df["species"] == category]
            df_averaged = pd.DataFrame({"species": [f"_{category}"]})
            for col in cols_to_mean:
                df_averaged[col] = df_categorised[col].mean()
            pplot.parallel_coordinates(ax=ax, frame=df_averaged, class_column="species", color=category_info[2],
                                       linewidth=3)

    def __plot_single_param(self, parameter: str, criteria_col:str, amount: int or None):
        df_iter = self.df_single_params[self.df_single_params["parameter"] == parameter]
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        parameter_values = self.df_original[parameter].unique().tolist()
        counter = 0
        for id, series_row in df_iter.iterrows():
            if counter == amount:
                break
            row = ast.literal_eval(series_row.to_dict()["combination"])
            criteria_values = list()
            for param_value in parameter_values:
                row[parameter] = param_value
                criteria_values.append(self.df_original.query(expr=self.__order_to_query(row))[criteria_col])
            ax.plot(parameter_values, criteria_values, label=id)
            counter += 1
        plt.title(f"Parameter: {parameter}"), plt.legend(), plt.xlabel(parameter), plt.ylabel(criteria_col), plt.grid()
        ax.set_xticks(self.df_original[parameter].unique())
        plt.savefig(self.dir_results+f"/{parameter}.png", dpi=180, bbox_inches="tight")
        plt.cla(), plt.clf()

    def __fill_categories(self, categories: dict[str:list]) -> dict:
        """Returns a dict. The keys are the percentage up to which the category (value) applies. The supplied
        categories need to be ascending!"""
        last_value = 0
        complete_categories = dict()
        for id, (category, interval) in enumerate(categories.items()):
            if interval[0] != last_value:
                complete_categories[interval[0]] = self._background_category
            complete_categories[interval[1]] = category
            if id == len(categories.keys()) - 1:
                if interval[1] != 100:
                    complete_categories[100] = self._background_category
            last_value = interval[1]
        return complete_categories

    def __get_single_param_combinations(self, parameter: str, criteria_col: str, non_parameter_cols: list[str]) -> \
            tuple[dict, list[float]]:
        """Check which unique values exist for the parameter col. Then check for all combinations of the other params if
         this combination exists for all unique values of the parameter col. If so, save the combination as a key in
         a dict and the indices of the rows as the value"""
        non_parameter_cols = list() if non_parameter_cols is None else non_parameter_cols
        df_filtered = self.df_original.drop([non_parameter_cols+[criteria_col]], axis=1)
        unique_values = self.df_original[parameter].unique().tolist()
        idx_species = dict()
        for id, series_row in df_filtered.iterrows():
            row = series_row.to_dict()
            row.pop(parameter)
            idx_current_species = list()
            exists_for_all_values = True
            for value in unique_values:
                query = self.__order_to_query((row|{parameter:value}))
                df_sliced = df_filtered.query(expr=query)
                if df_sliced.empty:
                    exists_for_all_values = False
                    break
                idx_current_species.append(df_sliced.index[0])
            df_filtered = df_filtered.drop(idx_current_species)
            if exists_for_all_values:
                idx_species[f"{row}"] = idx_current_species
        return idx_species, unique_values

    def __get_single_param_unique_combinations(self, parameter: str, criteria_col: str, non_parameter_cols: list[
        str]) -> dict:
        """Works like the function above but the values of the parameter do not have to be the same for each
        combination."""
        non_parameter_cols = list() if non_parameter_cols is None else non_parameter_cols
        df_filtered = self.df_original.drop([col for col in non_parameter_cols], axis=1)
        plot_combinations = dict()
        for id, series_row in df_filtered.iterrows():
            row = series_row.to_dict()
            row.pop(parameter), row.pop(criteria_col)
            plot_combinations[id] = {key: list() for key in [criteria_col, parameter]}
            query = self.__order_to_query(row)
            df_sliced = df_filtered.query(expr=query)
            for _, tmp_row in df_sliced.sort_values(by=parameter).iterrows():
                plot_combinations[id][criteria_col].append(tmp_row[criteria_col])
                plot_combinations[id][parameter].append(tmp_row[parameter])
            df_filtered = df_filtered.drop(df_sliced.index.tolist())
        return plot_combinations

    @staticmethod
    def __order_to_query(order: dict):
        query = str()
        for param, value in order.items():
            if type(value) == str or type(value) == list:
                query += f"{param}=='{value}' and "
            else:
                query += f"{param}=={value} and "
        return query[:-5]
