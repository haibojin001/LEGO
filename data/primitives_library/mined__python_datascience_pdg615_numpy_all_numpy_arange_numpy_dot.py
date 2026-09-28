# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg615::numpy.all+numpy.arange+numpy.dot
# name: numpy_sklearn_primitive
# summary: Uses numpy.all, numpy.arange, numpy.dot, numpy.eye across 2 repos
# anchor_symbols: ['numpy.all', 'numpy.arange', 'numpy.dot', 'numpy.eye', 'numpy.ones', 'numpy.tri', 'numpy.unique', 'numpy.vstack', 'numpy.zeros', 'sklearn.metrics.roc_auc_score']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/tasks/common_metric.py::auc_mu ---
def auc_mu(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
) -> float:
    """Compute multi-class metric AUC-Mu.

    We assume that confusion matrix full of ones, except diagonal elements.
    All diagonal elements are zeroes.
    By default, for averaging between classes scores we use simple mean.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Not used.
        class_weights: The between classes weight matrix. If ``None``,
            the standard mean will be used. It is expected to be a lower
            triangular matrix (diagonal is also full of zeroes).
            In position (i, j), i > j, there is a partial positive score
            between i-th and j-th classes. All elements must sum up to 1.

    Returns:
        Metric value.

    Note:
        Code was refactored from https://github.com/kleimanr/auc_mu/blob/master/auc_mu.py

    """
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"Expected y_pred to be np.ndarray, got: {type(y_pred)}")
    if not y_pred.ndim == 2:
        raise ValueError("Expected array with predictions be a 2-dimentional array")
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"Expected y_true to be np.ndarray, got: {type(y_true)}")
    if not y_true.ndim == 1:
        raise ValueError("Expected array with ground truths be a 1-dimentional array")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Expected number of samples in y_true and y_pred be same,"
            " got {} and {}, respectively".format(y_true.shape[0], y_pred.shape[0])
        )

    uniq_labels = np.unique(y_true)
    n_samples, n_classes = y_pred.shape

    if not np.all(uniq_labels == np.arange(n_classes)):
        raise ValueError("Expected classes encoded values 0, ..., N_classes-1")

    if class_weights is None:
        class_weights = np.tri(n_classes, k=-1)
        class_weights /= class_weights.sum()

    if not isinstance(class_weights, np.ndarray):
        raise TypeError(f"Expected class_weights to be np.ndarray, got: {type(class_weights)}")
    if not class_weights.ndim == 2:
        raise ValueError("Expected class_weights to be a 2-dimentional array")
    if not class_weights.shape == (n_classes, n_classes):
        raise ValueError(f"Expected class_weights size: {n_classes, n_classes}, got: {class_weights.shape}")
    # check sum?
    confusion_matrix = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    auc_full = 0.0

    for class_i in range(n_classes):
        preds_i = y_pred[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_pred[y_true == class_j]
            n_j = preds_j.shape[0]
            n = n_i + n_j
            tmp_labels = np.zeros((n,), dtype=np.int32)
            tmp_labels[n_i:] = 1
            tmp_pres = np.vstack((preds_i, preds_j))
            v = confusion_matrix[class_i, :] - confusion_matrix[class_j, :]
            scores = np.dot(tmp_pres, v)
            score_ij = roc_auc_score(tmp_labels, scores)
            auc_full += class_weights[class_i, class_j] * score_ij

    return auc_full

# --- from sberbank-ai-lab__LightAutoML::lightautoml/tasks/common_metric.py::auc_mu ---
def auc_mu(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
) -> float:
    """Compute multi-class metric AUC-Mu.

    We assume that confusion matrix full of ones, except diagonal elements.
    All diagonal elements are zeroes.
    By default, for averaging between classes scores we use simple mean.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Not used.
        class_weights: The between classes weight matrix. If ``None``,
            the standard mean will be used. It is expected to be a lower
            triangular matrix (diagonal is also full of zeroes).
            In position (i, j), i > j, there is a partial positive score
            between i-th and j-th classes. All elements must sum up to 1.

    Returns:
        Metric value.

    Note:
        Code was refactored from https://github.com/kleimanr/auc_mu/blob/master/auc_mu.py

    """
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("Expected y_pred to be np.ndarray, got: {}".format(type(y_pred)))
    if not y_pred.ndim == 2:
        raise ValueError("Expected array with predictions be a 2-dimentional array")
    if not isinstance(y_true, np.ndarray):
        raise TypeError("Expected y_true to be np.ndarray, got: {}".format(type(y_true)))
    if not y_true.ndim == 1:
        raise ValueError("Expected array with ground truths be a 1-dimentional array")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Expected number of samples in y_true and y_pred be same,"
            " got {} and {}, respectively".format(y_true.shape[0], y_pred.shape[0])
        )

    uniq_labels = np.unique(y_true)
    n_samples, n_classes = y_pred.shape

    if not np.all(uniq_labels == np.arange(n_classes)):
        raise ValueError("Expected classes encoded values 0, ..., N_classes-1")

    if class_weights is None:
        class_weights = np.tri(n_classes, k=-1)
        class_weights /= class_weights.sum()

    if not isinstance(class_weights, np.ndarray):
        raise TypeError("Expected class_weights to be np.ndarray, got: {}".format(type(class_weights)))
    if not class_weights.ndim == 2:
        raise ValueError("Expected class_weights to be a 2-dimentional array")
    if not class_weights.shape == (n_classes, n_classes):
        raise ValueError("Expected class_weights size: {}, got: {}".format((n_classes, n_classes), class_weights.shape))
    # check sum?
    confusion_matrix = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    auc_full = 0.0

    for class_i in range(n_classes):
        preds_i = y_pred[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_pred[y_true == class_j]
            n_j = preds_j.shape[0]
            n = n_i + n_j
            tmp_labels = np.zeros((n,), dtype=np.int32)
            tmp_labels[n_i:] = 1
            tmp_pres = np.vstack((preds_i, preds_j))
            v = confusion_matrix[class_i, :] - confusion_matrix[class_j, :]
            scores = np.dot(tmp_pres, v)
            score_ij = roc_auc_score(tmp_labels, scores)
            auc_full += class_weights[class_i, class_j] * score_ij

    return auc_full
