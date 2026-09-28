# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg416::numpy.floor
# name: numpy_primitive
# summary: Uses numpy.floor across 6 repos
# anchor_symbols: ['numpy.floor']
# observed in 6 repos: ['MatthewReid854__reliability', 'ModelOriented__DALEX', 'deepchecks__deepchecks', 'spotify__chartify', 'xorbitsai__xorbits']...

# --- from deepchecks__deepchecks::deepchecks/utils/numbers.py::round_sig ---
def round_sig(x: float, sig: int = 2):
    """Round a number to a given number of significant digits."""
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

# --- from ModelOriented__DALEX::python/dalex/dalex/predict_explanations/_break_down/utils.py::signif ---
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

# --- from ModelOriented__DALEX::python/dalex/dalex/predict_explanations/_shap/utils.py::signif ---
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

# --- from spotify__chartify::chartify/_core/plot.py::BasePlot._axis_format_precision ---
def _axis_format_precision(max_value, min_value):
        difference = abs(max_value - min_value)
        precision = abs(int(np.floor(np.log10(difference if difference else 1)))) + 1
        zeros = "".join(["0"] * precision)
        return "0,0.[{}]".format(zeros)

# --- from deepchecks__deepchecks::tests/tabular/checks/model_evaluation/weak_segments_performance_test.py::test_segment_performance_iris_score_per_sample ---
def test_segment_performance_iris_score_per_sample(iris_split_dataset_and_model):
    # Arrange
    _, val, model = iris_split_dataset_and_model

    score_per_sample = list(range(int(np.floor(val.n_samples / 2)))) + [1] * int(np.ceil(val.n_samples / 2))
    score_per_sample = pd.Series(score_per_sample, index=val.data.index)

    # Act
    result = WeakSegmentsPerformance(score_per_sample=score_per_sample).run(val, model)
    segments = result.value['weak_segments_list']

    # Assert
    assert_that(segments, any_of(has_length(5), has_length(6)))
    assert_that(segments.iloc[0, 0], close_to(1, 0.01))
    assert_that(segments.columns[0], equal_to('Average Score Per Sample'))

# --- from MatthewReid854__reliability::reliability/Utils.py::probability_plot_xyticks.customFormatter ---
def customFormatter(value, _):
        """
        Provides custom string formatting that is used for the xticks

        Parameters
        ----------
        value : int, float
            The value to be formatted

        Returns
        -------
        label : str
            The formatted string
        """
        if value == 0:
            label = "0"
        elif (
            abs(value) >= 10000 or abs(value) <= 0.0001
        ):  # small numbers and big numbers are formatted with scientific notation
            if value < 0:
                sign = "-"
                value *= -1
            else:
                sign = ""
            exponent = int(np.floor(np.log10(value)))
            multiplier = value / (10**exponent)
            if multiplier % 1 < 0.0000001:
                multiplier = int(multiplier)
            if multiplier == 1:
                label = str((r"$%s%s^{%d}$") % (sign, 10, exponent))
            else:
                label = str((r"$%s%g\times%s^{%d}$") % (sign, multiplier, 10, exponent))
        else:  # numbers between 0.0001 and 10000 are formatted without scientific notation
            label = str("{0:g}".format(value))
        return label

# --- from xorbitsai__xorbits::python/xorbits/_mars/tensor/stats/ks.py::_attempt_exact_2kssamp ---
def _attempt_exact_2kssamp(n1, n2, g, d, alternative):  # pragma: no cover
    """Attempts to compute the exact 2sample probability.

    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs

    Returns (success, d, probability)
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    saw_fp_error, prob = False, np.nan
    try:
        if alternative == "two-sided":
            if n1 == n2:
                prob = _compute_prob_outside_square(n1, h)
            else:
                prob = 1 - _compute_prob_inside_method(n1, n2, g, h)
        else:
            if n1 == n2:
                # prob = binom(2n, n-h) / binom(2n, n)
                # Evaluating in that form incurs roundoff errors
                # from special.binom. Instead calculate directly
                jrange = np.arange(h)
                prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
            else:
                num_paths = _count_paths_outside_method(n1, n2, g, h)
                bin = special.binom(n1 + n2, n1)  # pylint: disable=redefined-builtin
                if (
                    not np.isfinite(bin)
                    or not np.isfinite(num_paths)
                    or num_paths > bin
                ):
                    saw_fp_error = True
                else:
                    prob = num_paths / bin

    except FloatingPointError:
        saw_fp_error = True

    if saw_fp_error:
        return False, d, np.nan
    if not (0 <= prob <= 1):
        return False, d, prob
    return True, d, prob

# --- from zama-ai__concrete-ml::src/concrete/ml/onnx/onnx_impl_utils.py::compute_conv_output_dims ---
def compute_conv_output_dims(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
    ceil_mode: int,
) -> Tuple[int, ...]:
    """Compute the output shape of a pool or conv operation.

    See https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html for details
    on the computation of the output shape.

    Args:
        input_shape (Tuple[int, ...]): shape of the input to be padded as N x C x H x W
        kernel_shape (Tuple[int, ...]): shape of the conv or pool kernel, as Kh x Kw (or n-d)
        pads (Tuple[int, ...]): padding values following ONNX spec:
            dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end
            where in the 2-d case dim1 is H, dim2 is W
        strides (Tuple[int, ...]): strides for each dimension
        ceil_mode (int): set to 1 to use the `ceil` function to compute the output shape, as
            described in the PyTorch doc

    Returns:
        res (Tuple[int, ...]): shape of the output of a conv or pool operator with given parameters
    """

    assert_true(ceil_mode in {0, 1})

    height_out = (input_shape[2] + pads[0] + pads[2] - kernel_shape[0]) / strides[0] + 1
    width_out = (input_shape[3] + pads[1] + pads[3] - kernel_shape[1]) / strides[1] + 1

    if ceil_mode == 0:
        height_out = numpy.floor(height_out)
        width_out = numpy.floor(width_out)
    else:
        height_out = numpy.ceil(height_out)
        width_out = numpy.ceil(width_out)

    height_out = int(height_out)
    width_out = int(width_out)

    return (input_shape[0], input_shape[1], height_out, width_out)

# --- from xorbitsai__xorbits::python/xorbits/_mars/tensor/stats/ks.py::_calc_prob_2samp ---
def _calc_prob_2samp(d, n1, n2, alternative, mode):  # pragma: no cover
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N

    g = gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -mt.inf
    original_mode = mode
    if mode == "auto":
        mode = "exact" if max(n1, n2) <= MAX_AUTO_N else "asymp"
    elif mode == "exact":
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int_).max / n2g:
            mode = "asymp"
            warnings.warn(
                f"Exact ks_2samp calculation not possible with samples sizes "
                f"{n1} and {n2}. Switching to 'asymp'.",
                RuntimeWarning,
            )

    if mode == "exact":
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = "asymp"
            if original_mode == "exact":
                warnings.warn(
                    f"ks_2samp: Exact calculation unsuccessful. "
                    f"Switching to mode={mode}.",
                    RuntimeWarning,
                )

    if mode == "asymp":
        # The product n1*n2 is large.  Use Smirnov's asymptotic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == "two-sided":
            prob = distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
            prob = np.exp(expt)

    return np.clip(prob, 0, 1)

# --- from zama-ai__concrete-ml::src/concrete/ml/sklearn/tree_to_numpy.py::tree_values_preprocessing ---
def tree_values_preprocessing(
    onnx_model: onnx.ModelProto,
    framework: str,
    output_n_bits: int,
) -> QuantizedArray:
    """Pre-process tree values.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        output_n_bits (int): The number of bits of the output.

    Returns:
        QuantizedArray: Quantizer for the tree predictions.
    """
    q_y = QuantizedArray(
        n_bits=1, values=numpy.zeros(shape=(2,), dtype=numpy.float64), value_is_float=True
    )

    # Modify ONNX graph to fit in FHE
    for i, initializer in enumerate(onnx_model.graph.initializer):

        # All constants in our tree should be integers.
        # Tree thresholds can be rounded up or down (depending on the tree implementation)
        # while the final probabilities/regression values must be quantized.
        # We extract the value stored in each initializer node into the init_tensor.
        init_tensor = numpy_helper.to_array(initializer)
        if "weight_3" in initializer.name:
            # weight_3 is the prediction tensor, apply the required pre-processing
            q_y = preprocess_tree_predictions(init_tensor, output_n_bits)

            # Get the preprocessed tree predictions to replace the current (non-quantized)
            # values in the onnx_model.
            init_tensor = q_y.qvalues

        elif "bias_1" in initializer.name:
            if framework == "xgboost":
                # xgboost uses "<" (Less) operator thus we must round up.
                init_tensor = numpy.ceil(init_tensor)
            elif framework == "sklearn":
                # sklearn trees use "<=" (LessOrEqual) operator thus we must round down.
                init_tensor = numpy.floor(init_tensor)
        new_initializer = numpy_helper.from_array(init_tensor.astype(numpy.int64), initializer.name)
        onnx_model.graph.initializer[i].CopyFrom(new_initializer)

    return q_y
