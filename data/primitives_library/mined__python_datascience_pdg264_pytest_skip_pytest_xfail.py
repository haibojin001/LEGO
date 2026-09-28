# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg264::pytest.skip+pytest.xfail
# name: pytest_primitive
# summary: Uses pytest.skip, pytest.xfail across 2 repos
# anchor_symbols: ['pytest.skip', 'pytest.xfail']
# observed in 2 repos: ['modin-project__modin', 'vaexio__vaex']...

# --- from vaexio__vaex::tests/dataframe_protocol_test.py::test_smoke_get_chunks ---
def test_smoke_get_chunks(df_factory, n_chunks):
    if n_chunks is not None:
        pytest.xfail("get_chunks(n_chunks=...) doesn't work on already chunked columns")
    df = df_factory(x=[0])
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name('x')
    if isinstance(df["x"].values, pa.lib.ChunkedArray):
        pytest.skip("get_chunks() is slow/halts with chunked arrow arrays")
    interchange_col.get_chunks(n_chunks=n_chunks)

# --- from modin-project__modin::modin/tests/pandas/dataframe/test_binary.py::test_math_functions ---
def test_math_functions(other, axis, op, backend):
    data = test_data["float_nan_data"]
    if (op == "floordiv" or op == "rfloordiv") and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op == "rmod" and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op in ("mod", "rmod") and backend == "pyarrow":
        pytest.skip(reason="These functions are not implemented in pandas itself")
    eval_general(
        *create_test_dfs(data, backend=backend),
        lambda df: getattr(df, op)(other(df, axis), axis=axis),
    )

# --- from modin-project__modin::modin/tests/pandas/native_df_interoperability/test_binary.py::test_math_functions ---
def test_math_functions(other, axis, op, backend, df_mode_pair):
    data = test_data["float_nan_data"]
    if (op == "floordiv" or op == "rfloordiv") and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op == "rmod" and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op in ("mod", "rmod") and backend == "pyarrow":
        pytest.skip(reason="These functions are not implemented in pandas itself")

    eval_general_interop(
        data,
        backend,
        lambda df1, df2: getattr(df1, op)(other(df2, axis), axis=axis),
        df_mode_pair,
    )
