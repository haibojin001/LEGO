# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg616::numpy.arange+numpy.clip+numpy.ones_like
# name: numpy_primitive
# summary: Uses numpy.arange, numpy.clip, numpy.ones_like, numpy.zeros_like across 4 repos
# anchor_symbols: ['numpy.arange', 'numpy.clip', 'numpy.ones_like', 'numpy.zeros_like']
# observed in 4 repos: ['HunterMcGushion__hyperparameter_hunter', 'sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML', 'zama-ai__concrete-ml']...

# --- from sb-ai-lab__LightAutoML::lightautoml/tasks/losses/lgb_custom.py::lgb_f1_loss_multiclass ---
def lgb_f1_loss_multiclass(
    preds: np.ndarray, train_data: lgb.Dataset, clip: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """Custom loss for optimizing f1.

    Args:
        preds: Predctions.
        train_data: Dataset in LightGBM format.
        clip: Clump constant.

    Returns:
        Gradient, hessian.

    """
    y_true = train_data.get_label().astype(np.int32)
    preds = preds.reshape((y_true.shape[0], -1), order="F")
    # softmax
    preds = np.clip(softmax_ax1(preds), clip, 1 - clip)
    # make ohe
    y_ohe = np.zeros_like(preds)
    np.add.at(y_ohe, (np.arange(y_true.shape[0]), y_true), 1)
    # grad
    grad = (preds - y_ohe) * preds
    # hess
    hess = (1 - preds) * preds * np.clip((2 * preds - y_ohe), 1e-3, np.inf)
    # reshape back preds
    return grad.reshape((-1,), order="F"), hess.reshape((-1,), order="F")

# --- from sberbank-ai-lab__LightAutoML::lightautoml/tasks/losses/lgb_custom.py::lgb_f1_loss_multiclass ---
def lgb_f1_loss_multiclass(
    preds: np.ndarray, train_data: lgb.Dataset, clip: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """Custom loss for optimizing f1.

    Args:
        preds: Predctions.
        train_data: Dataset in LightGBM format.
        clip: Clump constant.

    Returns:
        Gradient, hessian.

    """
    y_true = train_data.get_label().astype(np.int32)
    preds = preds.reshape((y_true.shape[0], -1), order="F")
    # softmax
    preds = np.clip(softmax_ax1(preds), clip, 1 - clip)
    # make ohe
    y_ohe = np.zeros_like(preds)
    np.add.at(y_ohe, (np.arange(y_true.shape[0]), y_true), 1)
    # grad
    grad = (preds - y_ohe) * preds
    # hess
    hess = (1 - preds) * preds * np.clip((2 * preds - y_ohe), 1e-3, np.inf)
    # reshape back preds
    return grad.reshape((-1,), order="F"), hess.reshape((-1,), order="F")

# --- from HunterMcGushion__hyperparameter_hunter::tests/test_space/test_skopt_space.py::test_normalize ---
def test_normalize():
    # TODO: Refactor - Use PyTest
    a = Real(2.0, 30.0, transform="normalize")
    for i in range(50):
        check_limits(a.rvs(random_state=i), 2, 30)

    rng = np.random.RandomState(0)
    X = rng.randn(100)
    X = 28 * (X - X.min()) / (X.max() - X.min()) + 2

    # Check transformed values are in [0, 1]
    assert np.all(a.transform(X) <= np.ones_like(X))
    assert np.all(np.zeros_like(X) <= a.transform(X))

    # Check inverse transform
    assert_array_almost_equal(a.inverse_transform(a.transform(X)), X)

    # log-uniform prior
    a = Real(10 ** 2.0, 10 ** 4.0, prior="log-uniform", transform="normalize")
    for i in range(50):
        check_limits(a.rvs(random_state=i), 10 ** 2, 10 ** 4)

    rng = np.random.RandomState(0)
    X = np.clip(10 ** 3 * rng.randn(100), 10 ** 2.0, 10 ** 4.0)

    # Check transform
    assert np.all(a.transform(X) <= np.ones_like(X))
    assert np.all(np.zeros_like(X) <= a.transform(X))

    # Check inverse transform
    assert_array_almost_equal(a.inverse_transform(a.transform(X)), X)

    a = Integer(2, 30, transform="normalize")
    for i in range(50):
        check_limits(a.rvs(random_state=i), 2, 30)
    assert_array_equal(a.transformed_bounds, (0, 1))

    X = rng.randint(2, 31)
    # Check transformed values are in [0, 1]
    assert np.all(a.transform(X) <= np.ones_like(X))
    assert np.all(np.zeros_like(X) <= a.transform(X))

    # Check inverse transform
    X_orig = a.inverse_transform(a.transform(X))
    assert_equal(X_orig.dtype, "int64")
    assert_array_equal(X_orig, X)

# --- from zama-ai__concrete-ml::src/concrete/ml/onnx/ops_impl.py::numpy_slice ---
def numpy_slice(
    x: numpy.ndarray,
    starts: numpy.ndarray,
    ends: numpy.ndarray,
    axes: Optional[numpy.ndarray],
    steps: Optional[numpy.ndarray],
) -> Tuple[numpy.ndarray]:
    """Slice the input according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#slice-13

    Args:
        x (numpy.ndarray): input tensor to slice
        starts (numpy.ndarray): the starting indices, one for each axis to slice
        ends (numpy.ndarray): the ending indices, one for each axis to slice
        axes (numpy.ndarray): the axis indices, default is all axes
        steps (numpy.ndarray): the steps along each axis, defaults to 1

    Returns:
        result (Tuple[numpy.ndarray]): the slice(s) of the input tensor as a new tensor
    """

    slices = []
    if steps is None:
        steps = numpy.ones_like(starts)

    if axes is None:
        axes = numpy.arange(x.ndim)
        assert_true(
            starts.shape[0] == x.ndim and steps.shape[0] == x.ndim,
            "The Starts and Ends parameter of Slice must have the same "
            "number of elements as the number of axes of the input when the axes "
            f"parameter is None. Got starts with {starts.shape[0]} elements, ends "
            f"with {steps.shape[0]} elements, while the input has {x.ndim} dimensions.",
        )
    else:
        # Adjust negative axes
        axes = axes.copy()
        axes[axes < 0] += x.ndim
        assert_true(
            steps.shape[0] == starts.shape[0]
            and ends.shape[0] == starts.shape[0]
            and axes.shape[0] == starts.shape[0],
            "The Starts and Ends parameter of Slice must have the same "
            "number of elements as the axes parameter. Got starts with "
            f"{starts.shape[0]} elements, ends "
            f"with {steps.shape[0]} elements, while the axes had {axes.shape[0]} dimensions.",
        )

    # All negative values in starts[i] and ends[i] have dims[axes[i]] added to them,
    # where dims are the dimensions of input. Then start[axes[i]] is the adjusted starts[i]
    # is clamped into the range [0, dims[axes[i]]] for positive stepping and
    # [0, dims[axes[i]]-1] for negative stepping.

    # The clamping for the adjusted ends[i] depends on the sign of steps[i] and must
    # accommodate copying 0 through dims[axes[i]] elements,
    # so for positive stepping end[axes[i]] is clamped to [0, dims[axes[i]]],
    # while for negative stepping it is clamped to [-1, dims[axes[i]]-1].

    starts = starts.copy()
    ends = ends.copy()
    for k in range(starts.size):
        if starts[k] < 0:
            starts[k] += x.shape[axes[k]]
        if ends[k] < 0:
            ends[k] += x.shape[axes[k]]

        if steps[k] < 0:
            starts[k] = numpy.clip(starts[k], -x.shape[axes[k]] - 1, -1)
            ends[k] = numpy.clip(ends[k], -x.shape[axes[k]] - 1, -1)
        else:
            starts[k] = numpy.clip(starts[k], 0, x.shape[axes[k]] - 1)
            ends[k] = numpy.clip(ends[k], 0, x.shape[axes[k]])

    # Check there are no duplicates
    assert_true(
        len(numpy.unique(axes)) == len(axes), "Axes parameter to Slice contained duplicates"
    )

    # Initialize slices to take the whole input tensor
    slices = [slice(0, int(x.shape[axis]), 1) for axis in range(x.ndim)]

    for idx, axis in enumerate(axes):
        slices[axis] = slice(int(starts[idx]), int(ends[idx]), int(steps[idx]))

    return (x[tuple(slices)],)
