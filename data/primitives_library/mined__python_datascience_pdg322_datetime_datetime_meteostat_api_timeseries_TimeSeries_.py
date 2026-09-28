# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg322::datetime.datetime+meteostat.api.timeseries.TimeSeries+pandas.DataFrame
# name: datetime_meteostat_primitive
# summary: Uses datetime.datetime, meteostat.api.timeseries.TimeSeries, pandas.DataFrame, pandas.date_range across 2 repos
# anchor_symbols: ['datetime.datetime', 'meteostat.api.timeseries.TimeSeries', 'pandas.DataFrame', 'pandas.date_range']
# observed in 2 repos: ['alteryx__featuretools', 'meteostat__meteostat']...

# --- from alteryx__featuretools::featuretools/tests/primitive_tests/transform_primitive_tests/test_transform_primitive.py::test_age_two_years_quarterly ---
def test_age_two_years_quarterly():
    age = Age()
    dates = pd.Series(pd.date_range("2010-01-01", "2011-12-31", freq="Q"))
    ages = age(dates, time=datetime(2020, 2, 26))
    correct_ages = [9.915, 9.666, 9.414, 9.162, 8.915, 8.666, 8.414, 8.162]
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)

# --- from meteostat__meteostat::tests/unit/test_timeseries.py::TestTimeSeriesStartEnd.test_start_end_from_df_when_not_provided ---
def test_start_end_from_df_when_not_provided(self, stations_df):
        """When not provided, start/end should fall back to df bounds"""
        dates = pd.date_range("2024-01-15", "2024-01-20", freq="D")
        index = pd.MultiIndex.from_arrays(
            [["TEST"] * len(dates), dates, ["meteostat"] * len(dates)],
            names=["station", "time", "source"],
        )
        df = pd.DataFrame({Parameter.TEMP: range(len(dates))}, index=index)

        ts = TimeSeries(
            granularity=Granularity.DAILY,
            stations=stations_df,
            df=df,
        )

        assert ts.start == datetime(2024, 1, 15)
        assert ts.end == datetime(2024, 1, 20)

# --- from meteostat__meteostat::tests/unit/test_timeseries.py::TestTimeSeriesCompleteness.test_completeness_all_none_values ---
def test_completeness_all_none_values(self, stations_df):
        """completeness() with all None values should return 0.0"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        index = pd.MultiIndex.from_arrays(
            [["TEST"] * 5, dates, ["meteostat"] * 5],
            names=["station", "time", "source"],
        )
        df = pd.DataFrame({Parameter.TEMP: [None] * 5}, index=index)

        ts = TimeSeries(
            granularity=Granularity.DAILY,
            stations=stations_df,
            df=df,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        result = ts.completeness(Parameter.TEMP)
        assert result == 0.0
