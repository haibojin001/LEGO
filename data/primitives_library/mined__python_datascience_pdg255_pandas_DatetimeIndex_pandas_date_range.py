# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg255::pandas.DatetimeIndex+pandas.date_range
# name: pandas_primitive
# summary: Uses pandas.DatetimeIndex, pandas.date_range across 6 repos
# anchor_symbols: ['pandas.DatetimeIndex', 'pandas.date_range']
# observed in 6 repos: ['CamDavidsonPilon__lifetimes', 'alteryx__evalml', 'databricks__koalas', 'modin-project__modin', 'predict-idlab__plotly-resampler']...

# --- from alteryx__evalml::evalml/tests/conftest.py::uneven_end ---
def uneven_end():
    dates_1 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates_2 = pd.DatetimeIndex(["2009-03-26"])
    dates = dates_1.append(dates_2)
    return dates

# --- from alteryx__evalml::evalml/tests/conftest.py::missing_end ---
def missing_end():
    dates_1 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates_2 = pd.DatetimeIndex(["2009-05-01"])
    dates = dates_1.append(dates_2)
    return dates

# --- from modin-project__modin::modin/tests/pandas/test_io.py::test_to_period ---
def test_to_period():
    index = pandas.DatetimeIndex(
        pandas.date_range("2000", freq="h", periods=len(TEST_DATA["col1"]))
    )
    modin_df, pandas_df = create_test_dfs(TEST_DATA, index=index)
    df_equals(modin_df.to_period(), pandas_df.to_period())

# --- from modin-project__modin::modin/tests/pandas/test_io.py::test_df_to_dask ---
def test_df_to_dask():
    index = pandas.DatetimeIndex(
        pandas.date_range("2000", freq="h", periods=len(TEST_DATA["col1"]))
    )

    modin_df, pandas_df = create_test_dfs(TEST_DATA, index=index)

    dask_df = modin_df.modin.to_dask()
    df_equals(dask_df.compute(), pandas_df)

# --- from databricks__koalas::databricks/koalas/tests/indexes/test_datetime.py::DatetimeIndexTest.pidxs ---
def pidxs(self):
        return [
            pd.DatetimeIndex([0]),
            pd.DatetimeIndex(["2004-01-01", "2002-12-31", "2000-04-01"]),
        ] + [
            pd.date_range("2000-01-01", periods=3, freq=freq)
            for freq in (self.fixed_freqs + self.non_fixed_freqs)
        ]

# --- from stitchfix__hamilton::tests/resources/smoke_screen_module.py::spend_and_signups_source__pd ---
def spend_and_signups_source__pd(start_date: str, end_date: str) -> pd.DataFrame:
    """Yields `spend`, `signups`, and 'weeks' (the spine) in a single dataframe.
    All of this is obviously dummy data.
    """
    index = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date, freq="7d"))
    df_out = pd.DataFrame(data=dict(weeks=index, counts=list(range(len(index)))))
    df_out["spend"] = df_out.apply(lambda row: 100 + min((row.counts + 1) * 5, 250), axis=1)
    df_out["signups"] = df_out.apply(lambda row: 20 * min(row.counts + 1, 100), axis=1)
    return df_out[["spend", "signups", "weeks"]]

# --- from predict-idlab__plotly-resampler::tests/test_figure_resampler.py::test_set_hfx_tz_aware_series ---
def test_set_hfx_tz_aware_series():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01", "2020-01-02", freq="1s"
            ).tz_localize("Asia/Seoul")
        }
    )
    df["value"] = np.random.randn(len(df))

    fr = FigureResampler()
    fr.add_trace({}, hf_x=pd.Index(df.timestamp), hf_y=df.value)
    assert isinstance(fr.hf_data[0]["x"], pd.DatetimeIndex)
    # Now we set the pd.Series as hf_x
    fr.hf_data[0]["x"] = df.timestamp
    assert not isinstance(fr.hf_data[0]["x"], pd.DatetimeIndex)
    # perform an update
    out = fr._construct_update_data(
        {"xaxis.autorange": True, "xaxis.showspikes": False}
    )
    assert len(out) == 2
    # assert that the update was performed correctly
    assert isinstance(fr.hf_data[0]["x"], pd.DatetimeIndex)
    assert all(fr.hf_data[0]["x"] == pd.DatetimeIndex(df.timestamp))

# --- from CamDavidsonPilon__lifetimes::lifetimes/plotting.py::plot_history_alive ---
def plot_history_alive(model, t, transactions, datetime_col, freq="D", start_date=None, ax=None, **kwargs):
    """
    Draw a graph showing the probability of being alive for a customer in time.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    t: int
        the number of time units since the birth we want to draw the p_alive
    transactions: pandas DataFrame
        DataFrame containing the transactions history of the customer_id
    datetime_col: str
        The column in the transactions that denotes the datetime the purchase was made
    freq: str, optional
        Default 'D' for days. Other examples= 'W' for weekly
    start_date: datetime, optional
        Limit xaxis to start date
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    kwargs
        Passed into the matplotlib.pyplot.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if start_date is None:
        start_date = min(transactions[datetime_col])

    if ax is None:
        ax = plt.subplot(111)

    # Get purchasing history of user
    customer_history = transactions[[datetime_col]].copy()
    customer_history.index = pd.DatetimeIndex(customer_history[datetime_col])

    # Add transactions column
    customer_history["transactions"] = 1
    customer_history = customer_history.resample(freq).sum()

    # plot alive_path
    path = calculate_alive_path(model, transactions, datetime_col, t, freq)
    path_dates = pd.date_range(start=min(transactions[datetime_col]), periods=len(path), freq=freq)
    plt.plot(path_dates, path, "-", label="P_alive")

    # plot buying dates
    payment_dates = customer_history[customer_history["transactions"] >= 1].index
    plt.vlines(payment_dates.values, ymin=0, ymax=1, colors="r", linestyles="dashed", label="purchases")

    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(start_date, path_dates[-1])
    plt.legend(loc=3)
    plt.ylabel("P_alive")
    plt.title("History of P_alive")

    return ax
