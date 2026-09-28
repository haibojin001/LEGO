# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg795::numpy.any+sklearn.utils.multiclass.type_of_target
# name: numpy_sklearn_primitive
# summary: Uses numpy.any, sklearn.utils.multiclass.type_of_target across 2 repos
# anchor_symbols: ['numpy.any', 'sklearn.utils.multiclass.type_of_target']
# observed in 2 repos: ['ZhiningLiu1998__imbalanced-ensemble', 'sberbank-ai-lab__LightAutoML']...

# --- from ZhiningLiu1998__imbalanced-ensemble::imbens/utils/_validation.py::check_target_type ---
def check_target_type(y, indicate_one_vs_all=False):
    """Check the target types to be conform to the current samplers.

    The current samplers should be compatible with ``'binary'``,
    ``'multilabel-indicator'`` and ``'multiclass'`` targets only.

    Parameters
    ----------
    y : ndarray
        The array containing the target.

    indicate_one_vs_all : bool, default=False
        Either to indicate if the targets are encoded in a one-vs-all fashion.

    Returns
    -------
    y : ndarray
        The returned target.

    is_one_vs_all : bool, optional
        Indicate if the target was originally encoded in a one-vs-all fashion.
        Only returned if ``indicate_multilabel=True``.
    """
    type_y = type_of_target(y)
    if type_y == "multilabel-indicator":
        if np.any(y.sum(axis=1) > 1):
            raise ValueError(
                "Imbalanced-learn currently supports binary, multiclass and "
                "binarized encoded multiclasss targets. Multilabel and "
                "multioutput targets are not supported."
            )
        y = y.argmax(axis=1)
    else:
        y = column_or_1d(y)

    return (y, type_y == "multilabel-indicator") if indicate_one_vs_all else y

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/uplift/metrics.py::perfect_uplift_curve ---
def perfect_uplift_curve(y_true: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Calculate perfect curve.

    Method return curve's coordinates if the model is a perfect.
    Perfect model ranking:
        If type if 'y_true' is 'binary':
            1) Treatment = 1, Target = 1
            2) Treatment = 0, Target = 0
            3) Treatment = 1, Target = 0
            4) Treatment = 0, Target = 1

        If type if 'y_true' is 'continuous':
            Not implemented

    Args:
        y_true: Target values
        treatment: Treatment column

    Returns:
        perfect curve

    """
    if type_of_target(y_true) == "continuous" and np.any(y_true < 0.0):
        raise Exception("For a continuous target, the perfect curve is only available for non-negative values")

    if type_of_target(y_true) == "binary":
        perfect_control_score = (treatment == 0).astype(int) * (2 * (y_true != 1).astype(int) - 1)
        perfect_treatment_score = (treatment == 1).astype(int) * 2 * (y_true == 1).astype(int)
        perfect_uplift = perfect_treatment_score + perfect_control_score
    elif type_of_target(y_true) == "continuous":
        raise NotImplementedError("Can't calculate perfect curve for continuous target")
    else:
        raise RuntimeError("Only 'binary' and 'continuous' targets are available")

    return perfect_uplift
