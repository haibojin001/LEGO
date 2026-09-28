# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg560::pandas.DataFrame+sklearn.datasets.load_boston+sklearn.model_selection.train_test_split
# name: pandas_sklearn_primitive
# summary: Uses pandas.DataFrame, sklearn.datasets.load_boston, sklearn.model_selection.train_test_split across 2 repos
# anchor_symbols: ['pandas.DataFrame', 'sklearn.datasets.load_boston', 'sklearn.model_selection.train_test_split']
# observed in 2 repos: ['ClimbsRocks__auto_ml', 'rushter__heamy']...

# --- from rushter__heamy::tests/test_helpers.py::boston_dataset ---
def boston_dataset():
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
    return X_train, y_train, X_test, y_test

# --- from rushter__heamy::tests/test_pipeline.py::boston_dataset ---
def boston_dataset():
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
    return X_train, y_train, X_test, y_test

# --- from ClimbsRocks__auto_ml::auto_ml/utils.py::get_boston_dataset ---
def get_boston_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.2, random_state=42)
    return df_boston_train, df_boston_test

# --- from ClimbsRocks__auto_ml::tests/utils_testing.py::get_boston_regression_dataset ---
def get_boston_regression_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.33, random_state=42)
    return df_boston_train, df_boston_test
