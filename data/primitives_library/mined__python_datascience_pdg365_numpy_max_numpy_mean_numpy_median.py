# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg365::numpy.max+numpy.mean+numpy.median
# name: numpy_primitive
# summary: Uses numpy.max, numpy.mean, numpy.median, numpy.min across 4 repos
# anchor_symbols: ['numpy.max', 'numpy.mean', 'numpy.median', 'numpy.min', 'numpy.quantile']
# observed in 4 repos: ['ModelOriented__DALEX', 'awslabs__gluonts', 'sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/autots/base.py::TrendModel._detect_step ---
def _detect_step(self, x):
        x_min, x_max = tuple(
            np.quantile(x, [self.params["detect_step_quantile"], 1 - self.params["detect_step_quantile"]])
        )
        x_range = x_max - x_min
        window = self.params["detect_step_window"]
        diff = np.zeros(len(x) - 2 * window)
        for i in range(len(x) - 2 * window):
            diff[i] = np.median(x[i + window : i + 2 * window]) - np.median(x[i : i + window])
        diff = np.abs(diff) / x_range
        diff = np.concatenate((diff[0] * np.ones(window), diff, diff[-1] * np.ones(window)))
        return np.any(diff > self.params["detect_step_threshold"])

# --- from awslabs__gluonts::src/gluonts/mx/representation/global_relative_binning.py::GlobalRelativeBinning.initialize_from_array ---
def initialize_from_array(
        self, input_array: np.ndarray, ctx: mx.Context = get_mxnet_context()
    ):
        # Calculate bin centers and bin edges using linear or quantile binning.
        if self.is_quantile:
            bin_centers = np.quantile(
                input_array,
                np.linspace(0, self.quantile_scaling_limit, self.num_bins),
            )
            bin_centers = ensure_binning_monotonicity(bin_centers)
        else:
            has_negative_data = np.any(input_array < 0)
            low = -self.linear_scaling_limit if has_negative_data else 0
            high = self.linear_scaling_limit
            bin_centers = np.linspace(low, high, self.num_bins)
        bin_edges = bin_edges_from_bin_centers(bin_centers)

        # Store bin centers and edges since their are globally applicable to
        # all time series.
        with ctx:
            self.bin_edges.initialize()
            self.bin_centers.initialize()
        self.bin_edges.set_data(mx.nd.array(bin_edges))
        self.bin_centers.set_data(mx.nd.array(bin_centers))

# --- from ModelOriented__DALEX::python/dalex/dalex/predict_explanations/_ceteris_paribus/utils.py::calculate_variable_split ---
def calculate_variable_split(data,
                             variables,
                             grid_points,
                             variable_splits_type='uniform',
                             variable_splits_with_obs=False,
                             new_observation=None):
    """
    Calculate points for splitting the dataset

    :param data: dataset to split
    :param variables: variables to calculate ceteris paribus
    :param grid_points: how many points should split the dataset
    :param variable_splits_type: {'uniform', 'quantiles'}, optional way of calculating
        `variable_splits`. Set 'quantiles' for percentiles.
    :param variable_splits_with_obs: bool, optional add variable values of `new_observation`
        data to the `variable_splits`.
    :param new_observation: pd.DataFrame or np.ndarray, Observations for which predictions
        need to be explained.
    :return: dict, dictionary of split points for all variables
    """
    variable_splits = {}
    # grid points might be larger than the number of unique values
    probs = np.linspace(0, 1, grid_points)

    for variable in variables:
        variable_column = data.loc[:, variable]
        if pd.api.types.is_numeric_dtype(variable_column):
            if variable_splits_type == 'uniform':
                column_splits = np.linspace(np.min(variable_column),
                                            np.max(variable_column),
                                            grid_points)
            else:
                column_splits = np.unique(np.quantile(variable_column, probs))
            if variable_splits_with_obs:
                column_splits = np.concatenate((column_splits, new_observation.loc[:, variable]))
                column_splits = np.unique(column_splits)
                column_splits = np.sort(column_splits, kind='mergesort')

            variable_splits[variable] = column_splits
        else:
            variable_splits[variable] = variable_column.unique()

    return variable_splits

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDeco._describe_roles ---
def _describe_roles(self, train_data):

        # detect feature roles
        roles = self._model.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Numeric"]
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Category"]
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == "Datetime"]
        text_features = [feat_name for feat_name in roles if roles[feat_name].name == "Text"]

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["quantile_25"] = np.quantile(values, 0.25)
            item["average"] = np.mean(values)
            item["median"] = np.median(values)
            item["quantile_75"] = np.quantile(values, 0.75)
            item["max"] = np.max(values)
            numerical_features_df.append(item)
        self._numerical_features_table = list2table(numerical_features_df, {"float_format": "{:.2f}".format})

        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            value_counts = train_data[feature_name].value_counts(normalize=True)
            values = value_counts.index.values
            counts = value_counts.values
            item["Number of unique values"] = len(counts)
            item["Most frequent value"] = values[0]
            item["Occurance of most frequent"] = "{:.1f}%".format(100 * counts[0])
            item["Least frequent value"] = values[-1]
            item["Occurance of least frequent"] = "{:.1f}%".format(100 * counts[-1])
            categorical_features_df.append(item)
        self._categorical_features_table = list2table(categorical_features_df)

        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["max"] = np.max(values)
            item["base_date"] = self._model.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        self._datetime_features_table = list2table(datetime_features_df)

        # text roles
        text_features_df = []
        for feature_name in text_features:
            item = {"Feature name": feature_name}
            feature_length = train_data[feature_name].str.len()
            item["Amount of empty records"] = (feature_length == 0).sum(axis=0)
            item["Length of the shortest sentence"] = feature_length.min()
            item["Length of the longest sentence"] = feature_length.max()
            text_features_df.append(item)
        self._text_features_table = list2table(text_features_df)

# --- from sb-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDeco._describe_roles ---
def _describe_roles(self, train_data):

        # detect feature roles
        roles = self._model.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Numeric"]
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Category"]
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == "Datetime"]
        text_features = [feat_name for feat_name in roles if roles[feat_name].name == "Text"]

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = f"{train_data[feature_name].isna().sum() / train_data.shape[0]:.4f}"
            # check if column dtype is bool
            if train_data[feature_name].dtype == bool:
                values = train_data[feature_name].astype(float).dropna().values
            else:
                values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["quantile_25"] = np.quantile(values, 0.25)
            item["average"] = np.mean(values)
            item["median"] = np.median(values)
            item["quantile_75"] = np.quantile(values, 0.75)
            item["max"] = np.max(values)
            numerical_features_df.append(item)
        self._numerical_features_table = list2table(numerical_features_df, {"float_format": "{:.2f}".format})

        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = f"{train_data[feature_name].isna().sum() / train_data.shape[0]:.4f}"
            value_counts = train_data[feature_name].value_counts(normalize=True)
            values = value_counts.index.values
            counts = value_counts.values
            item["Number of unique values"] = len(counts)
            item["Most frequent value"] = values[0]
            item["Occurrence of most frequent"] = f"{100 * counts[0]:.1f}%"
            item["Least frequent value"] = values[-1]
            item["Occurrence of least frequent"] = f"{100 * counts[-1]:.1f}%"
            categorical_features_df.append(item)
        self._categorical_features_table = list2table(categorical_features_df)

        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = f"{train_data[feature_name].isna().sum() / train_data.shape[0]:.4f}"
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["max"] = np.max(values)
            item["base_date"] = self._model.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        self._datetime_features_table = list2table(datetime_features_df)

        # text roles
        text_features_df = []
        for feature_name in text_features:
            item = {"Feature name": feature_name}
            feature_length = train_data[feature_name].str.len()
            item["Amount of empty records"] = (feature_length == 0).sum(axis=0)
            item["Length of the shortest sentence"] = feature_length.min()
            item["Length of the longest sentence"] = feature_length.max()
            text_features_df.append(item)
        self._text_features_table = list2table(text_features_df)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDecoUplift._describe_roles ---
def _describe_roles(self, train_data):

        # detect feature roles
        # roles = self._model.reader._roles
        roles = self.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Numeric"]
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Category"]
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == "Datetime"]
        text_features = [feat_name for feat_name in roles if roles[feat_name].name == "Text"]

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["quantile_25"] = np.quantile(values, 0.25)
            item["average"] = np.mean(values)
            item["median"] = np.median(values)
            item["quantile_75"] = np.quantile(values, 0.75)
            item["max"] = np.max(values)
            numerical_features_df.append(item)
        if numerical_features_df == []:
            self._numerical_features_table = None
        else:
            self._numerical_features_table = pd.DataFrame(numerical_features_df).to_html(
                index=False, float_format="{:.2f}".format, justify="left"
            )
        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            value_counts = train_data[feature_name].value_counts(normalize=True)
            values = value_counts.index.values
            counts = value_counts.values
            item["Number of unique values"] = len(counts)
            item["Most frequent value"] = values[0]
            item["Occurance of most frequent"] = "{:.1f}%".format(100 * counts[0])
            item["Least frequent value"] = values[-1]
            item["Occurance of least frequent"] = "{:.1f}%".format(100 * counts[-1])
            categorical_features_df.append(item)
        if categorical_features_df == []:
            self._categorical_features_table = None
        else:
            self._categorical_features_table = pd.DataFrame(categorical_features_df).to_html(
                index=False, justify="left"
            )
        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["max"] = np.max(values)
            item["base_date"] = self.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        if datetime_features_df == []:
            self._datetime_features_table = None
        else:
            self._datetime_features_table = pd.DataFrame(datetime_features_df).to_html(index=False, justify="left")
        # text roles
        text_features_df = []
        for feature_name in text_features:
            item = {"Feature name": feature_name}
            feature_length = train_data[feature_name].str.len()
            item["Amount of empty records"] = (feature_length == 0).sum(axis=0)
            item["Length of the shortest sentence"] = feature_length.min()
            item["Length of the longest sentence"] = feature_length.max()
            text_features_df.append(item)
        self._text_features_table = list2table(text_features_df)
