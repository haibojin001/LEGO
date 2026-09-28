# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg323::numpy.isclose+pandas.to_datetime
# name: numpy_pandas_primitive
# summary: Uses numpy.isclose, pandas.to_datetime across 2 repos
# anchor_symbols: ['numpy.isclose', 'pandas.to_datetime']
# observed in 2 repos: ['alteryx__featuretools', 'lux-org__lux']...

# --- from alteryx__featuretools::featuretools/tests/primitive_tests/aggregation_primitive_tests/test_agg_primitives.py::test_trend_works_with_different_input_dtypes ---
def test_trend_works_with_different_input_dtypes():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    numeric = pd.Series([1, 2, 3])

    trend = Trend()
    dtypes = ["float64", "int64", "Int64"]

    for dtype in dtypes:
        actual = trend(numeric.astype(dtype), dates)
        assert np.isclose(actual, 1)

# --- from lux-org__lux::tests/test_interestingness.py::test_interestingness_deviation_nan ---
def test_interestingness_deviation_nan():
    import numpy as np

    dataset = [
        {"date": "2017-08-25", "category": "A", "value": 25.0},
        {"date": "2017-08-25", "category": "B", "value": 1.2},
        {"date": "2017-08-25", "category": "C", "value": 1.3},
        {"date": "2017-08-25", "category": "D", "value": 1.4},
        {"date": "2017-08-25", "category": "E", "value": 1.5},
        {"date": "2017-08-25", "category": "F", "value": 0.1},
        {"date": np.nan, "category": "C", "value": 0.2},
        {"date": np.nan, "category": "B", "value": 0.2},
        {"date": np.nan, "category": "F", "value": 0.3},
        {"date": np.nan, "category": "E", "value": 0.3},
        {"date": np.nan, "category": "D", "value": 0.4},
        {"date": np.nan, "category": "A", "value": 10.4},
        {"date": "2017-07-25", "category": "A", "value": 15.5},
        {"date": "2017-07-25", "category": "F", "value": 1.0},
        {"date": "2017-07-25", "category": "B", "value": 0.1},
    ]
    test = pd.DataFrame(dataset)
    from lux.vis.Vis import Vis

    test["date"] = pd.to_datetime(test["date"], format="%Y-%M-%d")
    test.set_data_type({"value": "quantitative"})

    vis = Vis(["date", "value", "category=A"], test)
    vis2 = Vis(["date", "value", "category=B"], test)
    from lux.interestingness.interestingness import interestingness

    smaller_diff_score = interestingness(vis, test)
    bigger_diff_score = interestingness(vis2, test)
    assert np.isclose(smaller_diff_score, 0.19, rtol=0.1)
    assert np.isclose(bigger_diff_score, 0.62, rtol=0.1)
    assert smaller_diff_score < bigger_diff_score
