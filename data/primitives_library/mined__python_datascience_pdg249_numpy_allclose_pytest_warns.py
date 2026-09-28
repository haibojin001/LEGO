# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg249::numpy.allclose+pytest.warns
# name: numpy_pytest_primitive
# summary: Uses numpy.allclose, pytest.warns across 2 repos
# anchor_symbols: ['numpy.allclose', 'pytest.warns']
# observed in 2 repos: ['HazyResearch__meerkat', 'cleanlab__cleanlab']...

# --- from cleanlab__cleanlab::tests/internal/test_numerics.py::TestSoftmax.test_shift ---
def test_shift(self, input_arr, expected_output):
        # Without shift, softmax overflows and gets a RuntimeWarning, but just returns nan
        with pytest.warns(RuntimeWarning):
            output_no_shift = softmax(input_arr, shift=False)
        assert np.isnan(output_no_shift).all()

        output_shift = softmax(input_arr, shift=True)
        assert np.allclose(output_shift, expected_output)

# --- from HazyResearch__meerkat::tests/meerkat/test_dataframe.py::test_json_io ---
def test_json_io(testbed, tmpdir):
    df = testbed.df
    filepath = os.path.join(tmpdir, "test.json")

    with pytest.warns():
        df.to_json(filepath)

    df2 = DataFrame.from_json(filepath, dtype=False)

    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in df2
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in df2
        else:
            assert name in df2
            if col.to_numpy().dtype == "object":
                assert np.all(df2[name].to_numpy() == col.to_numpy())
            else:
                assert np.allclose(df2[name].to_numpy(), col.to_numpy())
