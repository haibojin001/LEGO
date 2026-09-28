# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg325::datetime.datetime+numpy.datetime64
# name: datetime_numpy_primitive
# summary: Uses datetime.datetime, numpy.datetime64 across 2 repos
# anchor_symbols: ['datetime.datetime', 'numpy.datetime64']
# observed in 2 repos: ['alteryx__featuretools', 'feature-engine__feature_engine']...

# --- from alteryx__featuretools::featuretools/tests/entityset_tests/test_es.py::test_query_by_values_secondary_time_index ---
def test_query_by_values_secondary_time_index(es):
    end = np.datetime64(datetime(2011, 10, 1))
    all_instances = [0, 1, 2]
    result = es.query_by_values("customers", all_instances, time_last=end)

    for col in ["cancel_date", "cancel_reason"]:
        nulls = result.loc[all_instances][col].isnull() == [False, True, True]
        assert nulls.all(), "Some instance has data it shouldn't for column %s" % col

# --- from feature-engine__feature_engine::tests/test_selection/test_drop_high_psi_features.py::test_calculation_df_split_with_different_variable_types ---
def test_calculation_df_split_with_different_variable_types(df_mixed_types):
    """Test the split of the dataframe using different type of variables."""
    results = {}
    cut_offs = {}
    for split_col in df_mixed_types.columns:
        test = DropHighPSIFeatures(split_frac=0.5, split_col=split_col, variables="all")
        test.fit_transform(df_mixed_types)
        results[split_col] = test.psi_values_
        cut_offs[split_col] = test.cut_off_

    assert results["A"] == pytest.approx({"B": 0.0, "C": 0.1621860432432657}, 12)
    assert results["B"] == pytest.approx(
        {"A": 3.0375978817052403, "C": 8.515489752777954}, 12
    )
    assert results["C"] == pytest.approx({"A": 2.27819841127893, "B": 0.0}, 12)
    assert results["time"] == pytest.approx(
        {"A": 8.283089355027482, "B": 0.0, "C": 0.1621860432432657}, 12
    )

    expected_cut_offs = {
        "A": 9.5,
        "B": 1.5,
        "C": "B",
        "time": np.datetime64(datetime(2019, 1, 10)),
    }

    assert cut_offs == expected_cut_offs

    # Test when no dataframe with mixed data types when no split_col is provided.
    test = DropHighPSIFeatures(split_frac=0.5, variables="all")
    test.fit_transform(df_mixed_types)
    assert test.psi_values_ == pytest.approx(
        {"A": 8.283089355027482, "B": 0.0, "C": 0.1621860432432657}, 12
    )
