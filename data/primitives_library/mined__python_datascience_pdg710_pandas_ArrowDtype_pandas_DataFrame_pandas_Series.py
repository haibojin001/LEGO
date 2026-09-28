# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg710::pandas.ArrowDtype+pandas.DataFrame+pandas.Series
# name: pandas_pyarrow_primitive
# summary: Uses pandas.ArrowDtype, pandas.DataFrame, pandas.Series, pyarrow.string across 2 repos
# anchor_symbols: ['pandas.ArrowDtype', 'pandas.DataFrame', 'pandas.Series', 'pyarrow.string']
# observed in 2 repos: ['capitalone__datacompy', 'xorbitsai__xorbits']...

# --- from xorbitsai__xorbits::python/xorbits/_mars/dataframe/tests/test_utils.py::test_hash_utils ---
def test_hash_utils():
    df = pd.DataFrame({"foo": [1, 2, 3, 4], "bar": [5, 6, 7, 8]})
    res = hash_pandas_object(df)
    pd.testing.assert_series_equal(res, pd.util.hash_pandas_object(df))

    series = pd.Series(["a", "b", "c", "d"])
    res = hash_pandas_object(series)
    pd.testing.assert_series_equal(res, pd.util.hash_pandas_object(series))

    index = pd.Index([3.0, 4.0, 15.2, 22])
    res = hash_pandas_object(index)
    pd.testing.assert_series_equal(res, pd.util.hash_pandas_object(index))

    df = pd.DataFrame(
        {
            "foo": ["who", "is", "your", "daddy"],
            "bar": ["All", "hail", "Megatron", "!"],
        },
        dtype=pd.ArrowDtype(pa.string()),
    )
    res = hash_pandas_object(df, index=False)
    pd.testing.assert_series_equal(res, pd.util.hash_pandas_object(df, index=False))

# --- from capitalone__datacompy::tests/test_pandas.py::test_compare_empty_arrow_backed_join_keys ---
def test_compare_empty_arrow_backed_join_keys():
    """Regression for #514: empty Arrow-backed join keys must not crash merge."""
    import pyarrow as pa

    for arrow_type in (pa.string(), pa.int64()):
        dtype = pd.ArrowDtype(arrow_type)
        df1 = pd.DataFrame(
            {
                "key1": pd.Series([], dtype=dtype),
                "key2": pd.Series([], dtype=dtype),
                "value": pd.Series([], dtype=dtype),
            }
        )
        df2 = pd.DataFrame(
            {
                "key1": pd.Series([], dtype=dtype),
                "key2": pd.Series([], dtype=dtype),
                "value": pd.Series([], dtype=dtype),
            }
        )

        compare = PandasCompare(
            df1,
            df2,
            join_columns=["key1", "key2"],
            ignore_spaces=True,
            ignore_case=False,
        )

        assert len(compare.intersect_rows) == 0
        assert len(compare.df1_unq_rows) == 0
        assert len(compare.df2_unq_rows) == 0
