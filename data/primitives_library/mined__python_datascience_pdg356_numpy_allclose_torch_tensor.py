# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg356::numpy.allclose+torch.tensor
# name: numpy_torch_primitive
# summary: Uses numpy.allclose, torch.tensor across 2 repos
# anchor_symbols: ['numpy.allclose', 'torch.tensor']
# observed in 2 repos: ['Lightning-AI__torchmetrics', 'awslabs__gluonts']...

# --- from awslabs__gluonts::test/torch/model/test_torch_forecast.py::test_DistributionForecast ---
def test_DistributionForecast():
    forecast = DistributionForecast(
        distribution=Uniform(
            low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 2.0])
        ),
        start_date=START_DATE,
    )

    def percentile(value):
        return f"p{int(round(value * 100)):02d}"

    for quantile in QUANTILES:
        test_cases = [quantile, str(quantile), percentile(quantile)]
        for quant_pred in map(forecast.quantile, test_cases):
            expected = quantile * np.array([1.0, 2.0])
            assert np.allclose(
                quant_pred, expected
            ), f"Expected {percentile(quantile)} quantile {quantile}. Obtained {quant_pred}."

    pred_length = 2
    assert forecast.prediction_length == pred_length
    assert len(forecast.index) == pred_length
    assert forecast.index[0] == START_DATE

# --- from Lightning-AI__torchmetrics::tests/unittests/classification/test_calibration_error.py::test_corner_case_due_to_dtype ---
def test_corner_case_due_to_dtype():
    """Test that metric works with edge case where the precision is really important for the right result.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/1907

    """
    preds = torch.tensor(
        [0.9000, 0.9000, 0.9000, 0.9000, 0.9000, 0.8000, 0.8000, 0.0100, 0.3300, 0.3400, 0.9900, 0.6100],
        dtype=torch.float64,
    )
    target = torch.tensor([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0])

    assert np.allclose(
        ECE(99).measure(preds.numpy(), target.numpy()), binary_calibration_error(preds, target, n_bins=99)
    ), "The metric should be close to the netcal implementation"
    assert np.allclose(
        ECE(100).measure(preds.numpy(), target.numpy()), binary_calibration_error(preds, target, n_bins=100)
    ), "The metric should be close to the netcal implementation"

# --- from Lightning-AI__torchmetrics::tests/unittests/bases/test_metric.py::test_add_state ---
def test_add_state():
    """Test that add state method works as expected."""
    metric = DummyMetric()

    metric.add_state("a", tensor(0), "sum")
    assert metric._reductions["a"](tensor([1, 1])) == 2

    metric.add_state("b", tensor(0), "mean")
    assert np.allclose(metric._reductions["b"](tensor([1.0, 2.0])).numpy(), 1.5)

    metric.add_state("c", tensor(0), "cat")
    assert metric._reductions["c"]([tensor([1]), tensor([1])]).shape == (2,)

    with pytest.raises(ValueError, match="`dist_reduce_fx` must be callable or one of .*"):
        metric.add_state("d1", tensor(0), "xyz")

    with pytest.raises(ValueError, match="`dist_reduce_fx` must be callable or one of .*"):
        metric.add_state("d2", tensor(0), 42)

    with pytest.raises(ValueError, match="state variable must be a tensor or any empty list .*"):
        metric.add_state("d3", [tensor(0)], "sum")

    with pytest.raises(ValueError, match="state variable must be a tensor or any empty list .*"):
        metric.add_state("d4", 42, "sum")

    def custom_fx(_):
        return -1

    metric.add_state("e", tensor(0), custom_fx)
    assert metric._reductions["e"](tensor([1, 1])) == -1
