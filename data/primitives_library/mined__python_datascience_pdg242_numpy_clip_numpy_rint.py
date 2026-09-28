# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg242::numpy.clip+numpy.rint
# name: numpy_primitive
# summary: Uses numpy.clip, numpy.rint across 2 repos
# anchor_symbols: ['numpy.clip', 'numpy.rint']
# observed in 2 repos: ['cleanlab__cleanlab', 'zama-ai__concrete-ml']...

# --- from cleanlab__cleanlab::cleanlab/experimental/label_issues_batched.py::LabelInspector.get_num_issues ---
def get_num_issues(self, silent: bool = False) -> int:
        """
        Fetches already-computed estimate of the number of label issues in the data seen so far
        in the same format as: :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.

        Note: The estimated number of issues may differ from :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`
        by 1 due to rounding differences.

        Returns
        -------
        num_issues : int
          The estimated number of examples with label issues in the data seen so far.
        """
        if self.examples_processed_quality < 1:
            raise ValueError(
                "Have not evaluated any labels yet. Call `score_label_quality()` first."
            )
        else:
            if self.verbose and not silent:
                print(
                    f"Total number of examples whose labels have been evaluated: {self.examples_processed_quality}"
                )
            if self.off_diagonal_calibrated:
                calibrated_prune_counts = (
                    self.prune_counts
                    * self.class_counts
                    / np.clip(self.normalization, a_min=CLIPPING_LOWER_BOUND, a_max=None)
                )  # avoid division by 0
                return np.rint(np.sum(calibrated_prune_counts)).astype("int")
            else:  # not calibrated
                return self.prune_count

# --- from zama-ai__concrete-ml::src/concrete/ml/onnx/ops_impl.py::numpy_brevitas_quant ---
def numpy_brevitas_quant(
    x: numpy.ndarray,
    scale: float,
    zero_point: float,
    bit_width: int,
    *,
    rounding_mode: str = "ROUND",
    signed: int = 1,
    narrow: int = 0,
):
    """Quantize according to Brevitas uniform quantization.

    Args:
        x (numpy.ndarray): Tensor to be quantized
        scale (float): Quantizer scale
        zero_point (float): Quantizer zero-point
        bit_width (int): Number of bits of the integer representation
        rounding_mode (str): Rounding mode (default and only accepted option is "ROUND")
        signed (int): Whether this op quantizes to signed integers (default 1),
        narrow (int): Whether this op quantizes to a narrow range of integers
            e.g., [-2**n_bits-1 .. 2**n_bits-1] (default 0),

    Returns:
        result (numpy.ndarray): Tensor with float quantized values
    """

    assert_true(rounding_mode == "ROUND", "Only rounding quantization is supported for Brevitas")
    assert_true(signed in (1, 0), "Signed flag in Brevitas quantizer must be 0/1")
    assert_true(narrow in (1, 0), "Narrow range flag in Brevitas quantizer must be 0/1")

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4544
    # Remove this workaround when brevitas export is fixed
    if signed == 0 and narrow == 1:
        signed = 1
        narrow = 0

    assert_false(
        signed == 0 and narrow == 1,
        "Can not use narrow range for non-signed Brevitas quantizers",
    )

    # Compute the re-scaled values
    y = x / scale
    y = y + zero_point

    # Clip the values to the correct range
    min_int_val = min_int(signed, narrow, bit_width)
    max_int_val = max_int(signed, narrow, bit_width)
    y = numpy.clip(y, min_int_val, max_int_val)

    # Quantize to produce integers representing the float quantized values
    if utils.QUANT_ROUND_LIKE_ROUND_PBS:
        y = numpy.floor(y + 0.5)
    else:
        y = numpy.rint(y)

    # Compute quantized floating point values
    y = (y - zero_point) * scale

    return (y,)
