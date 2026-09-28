# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg811::datacompy.polars.columns_equal+polars.Array+polars.DataFrame
# name: datacompy_polars_primitive
# summary: Uses datacompy.polars.columns_equal, polars.Array, polars.DataFrame, polars.Series across 2 repos
# anchor_symbols: ['datacompy.polars.columns_equal', 'polars.Array', 'polars.DataFrame', 'polars.Series']
# observed in 2 repos: ['capitalone__datacompy', 'stitchfix__hamilton']...

# --- from stitchfix__hamilton::examples/polars/my_functions.py::base_df ---
def base_df(base_df_location: str) -> pl.DataFrame:
    """Loads base dataframe of data.

    :param base_df_location: just showing that we could load this from a file...
    :return:
    """
    return pl.DataFrame(
        {
            "signups": pl.Series([1, 10, 50, 100, 200, 400]),
            "spend": pl.Series([10, 10, 20, 40, 40, 50]),
        }
    )

# --- from capitalone__datacompy::tests/test_polars.py::test_columns_equal_arrays ---
def test_columns_equal_arrays():
    # all equal
    df1 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.Array(pl.Int64, 1)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.Array(pl.Int64, 1)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert actual.explode().all()

    # all mismatch
    df1 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.Array(pl.Int64, 1)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[2], [3], [4], [5], [6]]},
        schema={"array_col": pl.Array(pl.Int64, 1)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert not actual.explode().all()

    # some equal
    df1 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.Array(pl.Int64, 1)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[1], [1], [3], [4], [5]]},
        schema={"array_col": pl.Array(pl.Int64, 1)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert (actual.explode() == pl.Series([True, False, True, True, True])).all()

    # empty
    df1 = pl.DataFrame(
        {"array_col": [[], [], [], [], []]},
        schema={"array_col": pl.Array(pl.Int64, 0)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[], [], [], [], []]},
        schema={"array_col": pl.Array(pl.Int64, 0)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert actual.explode().all()

# --- from capitalone__datacompy::tests/test_polars.py::test_columns_equal_lists ---
def test_columns_equal_lists():
    # all equal
    df1 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.List(pl.Int64)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.List(pl.Int64)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert actual.all()

    # all mismatch
    df1 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.List(pl.Int64)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[2], [3], [4], [5], [6]]},
        schema={"array_col": pl.List(pl.Int64)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert not actual.all()

    # some equal
    df1 = pl.DataFrame(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema={"array_col": pl.List(pl.Int64)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[1], [1], [3], [4], [5]]},
        schema={"array_col": pl.List(pl.Int64)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert (actual == pl.Series([True, False, True, True, True])).all()

    # different shapes
    df1 = pl.DataFrame(
        {
            "array_col": [
                [],
                [np.nan],
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 2, 3],
            ]
        },
        schema={"array_col": pl.List(pl.Float64)},
    )
    df2 = pl.DataFrame(
        {
            "array_col": [
                [],
                [np.nan],
                [1, 2, 3],
                [1, 3],
                [2, 3],
                [1, 2],
            ]
        },
        schema={"array_col": pl.List(pl.Float64)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert (actual == pl.Series([True, True, False, True, True, False])).all()

    # empty
    df1 = pl.DataFrame(
        {"array_col": [[], [], [], [], []]},
        schema={"array_col": pl.Array(pl.Int64, 0)},
    )
    df2 = pl.DataFrame(
        {"array_col": [[], [], [], [], []]},
        schema={"array_col": pl.Array(pl.Int64, 0)},
    )
    actual = columns_equal(df1["array_col"], df2["array_col"])
    assert actual.all()
