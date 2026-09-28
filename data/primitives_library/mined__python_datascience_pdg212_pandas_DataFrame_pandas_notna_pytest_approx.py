# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg212::pandas.DataFrame+pandas.notna+pytest.approx
# name: pandas_pytest_primitive
# summary: Uses pandas.DataFrame, pandas.notna, pytest.approx across 2 repos
# anchor_symbols: ['pandas.DataFrame', 'pandas.notna', 'pytest.approx']
# observed in 2 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'meteostat__meteostat']...

# --- from meteostat__meteostat::tests/unit/test_lapserate.py::TestLapseRateNaNHandling.test_valid_values_are_adjusted ---
def test_valid_values_are_adjusted(self):
        """
        Non-NaN temperature values should be adjusted by lapse rate.
        """
        df = pd.DataFrame(
            {
                Parameter.TEMP: [20.0, 22.0],
                "elevation": [100, 100],
            }
        )

        result = apply_lapse_rate(df.copy(), elevation=200, lapse_rate=6.5)

        # Values should be adjusted (not NaN)
        assert pd.notna(result[Parameter.TEMP].iloc[0])
        assert pd.notna(result[Parameter.TEMP].iloc[1])

        # Adjustment: (6.5 / 1000) * (100 - 200) = -0.65
        # So temps should decrease by 0.65
        assert result[Parameter.TEMP].iloc[0] == pytest.approx(19.4, abs=0.1)
        assert result[Parameter.TEMP].iloc[1] == pytest.approx(21.4, abs=0.1)

# --- from Data-Centric-AI-Community__fg-data-profiling::tests/backends/spark_backend/test_descriptions_spark.py::test_describe_spark_df ---
def test_describe_spark_df(
    column,
    describe_data,
    expected_results,
    summarizer_spark,
    typeset,
    spark_session,
):
    cfg = SparkSettings()

    # disable correlations for description test
    cfg.correlations["pearson"].calculate = False
    cfg.correlations["spearman"].calculate = False

    if column == "mixed":
        describe_data[column] = [str(i) for i in describe_data[column]]
    elif column == "bool_tf_with_nan":
        describe_data[column] = [
            True if i else False for i in describe_data[column]  # noqa: SIM210
        ]
    pdf = pd.DataFrame({column: describe_data[column]})  # Convert to Pandas DataFrame
    # Ensure NaNs are replaced with None (Spark does not support NaN in non-float columns)
    pdf = pdf.where(pd.notna(pdf), None)

    sdf = spark_session.createDataFrame(pdf)

    results = describe(cfg, sdf, summarizer_spark, typeset)

    assert {
        "analysis",
        "time_index_analysis",
        "table",
        "variables",
        "scatter",
        "correlations",
        "missing",
        "package",
        "sample",
        "duplicates",
        "alerts",
    } == set(asdict(results).keys()), "Not in results"
    # Loop over variables
    for k, v in expected_results[column].items():
        if v == check_is_NaN:
            # test_condition should be True if column not in results, or the result is a nan value
            test_condition = k not in results.variables[column] or pd.isna(
                results.variables[column].get(k, np.nan)
            )
        elif isinstance(v, float):
            test_condition = (
                pytest.approx(v, nan_ok=True) == results.variables[column][k]
            )
        else:
            test_condition = v == results.variables[column][k]

        assert (
            test_condition
        ), f"Value `{results.variables[column][k]}` for key `{k}` in column `{column}` is not check_is_NaN"
