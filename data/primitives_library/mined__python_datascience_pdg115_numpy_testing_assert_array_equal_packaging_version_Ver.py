# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg115::numpy.testing.assert_array_equal+packaging.version.Version+pytest.skip
# name: numpy_packaging_primitive
# summary: Uses numpy.testing.assert_array_equal, packaging.version.Version, pytest.skip across 2 repos
# anchor_symbols: ['numpy.testing.assert_array_equal', 'packaging.version.Version', 'pytest.skip']
# observed in 2 repos: ['microsoft__nni', 'modin-project__modin']...

# --- from microsoft__nni::test/ut/nas/profiler/conftest.py::skip_for_legacy_pytorch ---
def skip_for_legacy_pytorch():
    import torch
    if Version(torch.__version__) < Version('1.11.0'):
        pytest.skip('PyTorch version is too old, skip this test.')

# --- from modin-project__modin::modin/tests/pandas/dataframe/test_default.py::test___array__ ---
def test___array__(data, copy_kwargs, get_array, get_array_name):
    if (
        get_array_name == "np.array"
        and Version(np.__version__) < Version("2")
        and "copy" in copy_kwargs
        and copy_kwargs["copy"] is None
    ):
        pytest.skip(reason="np.array does not support copy=None before numpy 2.0")
    assert_array_equal(*(get_array(df, copy_kwargs) for df in create_test_dfs(data)))

# --- from modin-project__modin::modin/tests/pandas/test_series.py::test___array__ ---
def test___array__(data, copy_kwargs, get_array, get_array_name):
    if (
        get_array_name == "np.array"
        and Version(np.__version__) < Version("2")
        and "copy" in copy_kwargs
        and copy_kwargs["copy"] is None
    ):
        pytest.skip(reason="np.array does not support copy=None before numpy 2.0")
    assert_array_equal(*(get_array(df, copy_kwargs) for df in create_test_series(data)))
