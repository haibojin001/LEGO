# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg278::numpy.round+sklearn.metrics.roc_auc_score+sklearn.utils.check_consistent_length
# name: numpy_sklearn_primitive
# summary: Uses numpy.round, sklearn.metrics.roc_auc_score, sklearn.utils.check_consistent_length, sklearn.utils.column_or_1d across 2 repos
# anchor_symbols: ['numpy.round', 'sklearn.metrics.roc_auc_score', 'sklearn.utils.check_consistent_length', 'sklearn.utils.column_or_1d']
# observed in 2 repos: ['yzhao062__combo', 'yzhao062__pyod']...

# --- from yzhao062__combo::combo/utils/data.py::evaluate_print ---
def evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include accuracy, roc, and F1 score

    Parameters
    ----------
    clf_name : str
        The name of the estimator.

    y : list or numpy array of shape (n_samples,)
        The ground truth.

    y_pred : list or numpy array of shape (n_samples,)
        The raw scores as returned by a fitted model.

    """

    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    print('{clf_name} Accuracy:{acc}, ROC:{roc}, F1:{f1}'.format(
        clf_name=clf_name,
        acc=np.round(accuracy_score(y, y_pred), decimals=4),
        roc=np.round(roc_auc_score(y, y_pred), decimals=4),
        f1=np.round(f1_score(y, y_pred), decimals=4)))

# --- from yzhao062__pyod::pyod/utils/data.py::evaluate_print ---
def evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include ROC and Precision @ n

    Parameters
    ----------
    clf_name : str
        The name of the detector.

    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    """

    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    print('{clf_name} ROC:{roc}, precision @ rank n:{prn}'.format(
        clf_name=clf_name,
        roc=np.round(roc_auc_score(y, y_pred), decimals=4),
        prn=np.round(precision_n_scores(y, y_pred), decimals=4)))

# --- from yzhao062__pyod::pyod/utils/data.py::check_consistent_shape ---
def check_consistent_shape(X_train, y_train, X_test, y_test, y_train_pred,
                           y_test_pred):
    """Internal shape to check input data shapes are consistent.

    Parameters
    ----------
    X_train : numpy array of shape (n_samples, n_features)
        The training samples.

    y_train : list or array of shape (n_samples,)
        The ground truth of training samples.

    X_test : numpy array of shape (n_samples, n_features)
        The test samples.

    y_test : list or array of shape (n_samples,)
        The ground truth of test samples.

    y_train_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the training samples.

    y_test_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the test samples.

    Returns
    -------
    X_train : numpy array of shape (n_samples, n_features)
        The training samples.

    y_train : list or array of shape (n_samples,)
        The ground truth of training samples.

    X_test : numpy array of shape (n_samples, n_features)
        The test samples.

    y_test : list or array of shape (n_samples,)
        The ground truth of test samples.

    y_train_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the training samples.

    y_test_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the test samples.
    """

    # check input data shapes are consistent
    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)

    y_test_pred = column_or_1d(y_test_pred)
    y_train_pred = column_or_1d(y_train_pred)

    check_consistent_length(y_train, y_train_pred)
    check_consistent_length(y_test, y_test_pred)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train {0} and X_test {1} have different number "
                         "of features.".format(X_train.shape, X_test.shape))

    return X_train, y_train, X_test, y_test, y_train_pred, y_test_pred
