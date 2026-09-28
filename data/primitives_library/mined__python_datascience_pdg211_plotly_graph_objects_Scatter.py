# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg211::plotly.graph_objects.Scatter
# name: plotly_primitive
# summary: Uses plotly.graph_objects.Scatter across 11 repos
# anchor_symbols: ['plotly.graph_objects.Scatter']
# observed in 11 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'ModelOriented__DALEX', 'alteryx__evalml', 'deepchecks__deepchecks', 'encord-team__encord-active']...

# --- from predict-idlab__plotly-resampler::tests/test_figure_resampler.py::test_add_scatter_trace_no_x ---
def test_add_scatter_trace_no_x():
    fig = FigureResampler(go.Figure(), default_n_shown_samples=1000)

    # no x data
    fig.add_trace(go.Scatter(y=[2, 1, 4, 3], name="s1"))
    fig.add_trace(go.Scatter(name="s2"), hf_y=[2, 1, 4, 3])

# --- from predict-idlab__plotly-resampler::tests/test_figurewidget_resampler.py::test_add_scatter_trace_no_x ---
def test_add_scatter_trace_no_x():
    fig = FigureWidgetResampler(go.Figure(), default_n_shown_samples=1000)

    # no x data
    fig.add_trace(go.Scatter(y=[2, 1, 4, 3], name="s1"))
    fig.add_trace(go.Scatter(name="s2"), hf_y=[2, 1, 4, 3])

# --- from Data-Centric-AI-Community__fg-data-profiling::tests/unit/test_time_series.py::sample_ts_df ---
def sample_ts_df():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": np.sin(np.arange(100) * np.pi / 180)
            + np.random.normal(0, 0.1, 100),
            "trend": np.arange(100) * 0.1,
            "category": ["A", "B"] * 50,
        }
    )

# --- from alteryx__evalml::evalml/tests/component_tests/decomposer_tests/test_decomposer.py::test_set_time_index ---
def test_set_time_index(decomposer_child_class):
    x = np.arange(0, 2 * np.pi, 0.01)
    dts = pd.date_range(datetime.today(), periods=len(x))
    X = pd.DataFrame({"x": x})
    X = X.set_index(dts)
    y = pd.Series(np.sin(x))

    assert isinstance(y.index, pd.RangeIndex)

    decomposer = decomposer_child_class()
    y_time_index = decomposer._set_time_index(X, y)
    assert isinstance(y_time_index.index, pd.DatetimeIndex)

# --- from starpig1129__DATAGEN::src/core/mcp_manager.py::reset_mcp_manager ---
def reset_mcp_manager() -> None:
    """Reset the MCPManager singleton.
    
    Useful for testing or when reconfiguration is needed.
    """
    global _default_manager
    if _default_manager is not None:
        # Try to cleanup connections
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
                
            if loop and not loop.is_running():
                loop.run_until_complete(_default_manager.close_all())
            elif not loop:
                asyncio.run(_default_manager.close_all())
        except Exception:
            pass
    _default_manager = None

# --- from violit-dev__violit::src/violit/widgets/chart_widgets.py::_plotly_json_dumps._normalize ---
def _normalize(value):
        if isinstance(value, np.ndarray):
            return [_normalize(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            if set(value.keys()) == {"dtype", "bdata"}:
                try:
                    array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
                    return [_normalize(item) for item in array.tolist()]
                except Exception:
                    return value
            return {key: _normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        if isinstance(value, tuple):
            return [_normalize(item) for item in value]
        return value

# --- from violit-dev__violit::src/violit/app.py::App._schedule_scoped_state_flush ---
def _schedule_scoped_state_flush(
        self,
        view_keys: Set[tuple[str, str]],
        *,
        exclude_current: bool = False,
    ) -> None:
        if not view_keys:
            return

        coroutine = self._flush_scoped_state_views_async(set(view_keys), exclude_current=exclude_current)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            running_loop.create_task(coroutine)
            return

        main_loop = getattr(self, '_main_loop', None)
        if main_loop is not None and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(coroutine, main_loop)
            return

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coroutine)
        finally:
            loop.close()

# --- from alteryx__evalml::evalml/tests/component_tests/test_arima_regressor.py::test_different_time_units_out_of_sample ---
def test_different_time_units_out_of_sample(
    freq_str,
    freq_num,
    sktime_arima,
    forecasting,
):
    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.base import ForecastingHorizon

    datetime_ = pd.date_range("1/1/1870", periods=20, freq=freq_num + freq_str)

    X = pd.DataFrame(range(20), index=datetime_)
    y = pd.Series(np.sin(np.linspace(-8 * np.pi, 8 * np.pi, 20)), index=datetime_)

    fh_ = ForecastingHorizon([i + 1 for i in range(len(y[15:]))], is_relative=True)

    a_clf = AutoARIMA(maxiter=10)
    clf = a_clf.fit(X=X[:15], y=y[:15])
    y_pred_sk = clf.predict(fh=fh_, X=X[15:])

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X[:15], y=y[:15])
    y_pred = m_clf.predict(X=X[15:])
    assert m_clf._component_obj.d is None

    np.testing.assert_almost_equal(y_pred_sk.values, y_pred.values)
    assert y_pred.index.equals(X[15:].index)

# --- from xorbitsai__xorbits::python/xorbits/_mars/utils.py::patch_asyncio_task_create_time ---
def patch_asyncio_task_create_time():  # pragma: no cover
    new_loop = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        new_loop = True
    loop_class = loop.__class__
    # Save raw loop_class.create_task and make multiple apply idempotent
    loop_create_task = getattr(
        patch_asyncio_task_create_time, "loop_create_task", loop_class.create_task
    )
    patch_asyncio_task_create_time.loop_create_task = loop_create_task

    def new_loop_create_task(*args, **kwargs):
        task = loop_create_task(*args, **kwargs)
        task.__mars_asyncio_task_create_time__ = time.time()
        return task

    if loop_create_task is not new_loop_create_task:
        loop_class.create_task = new_loop_create_task
    if not new_loop and loop.create_task is not new_loop_create_task:
        loop.create_task = functools.partial(new_loop_create_task, loop)

# --- from encord-team__encord-active::src/encord_active/lib/charts/data_quality_summary.py::create_outlier_distribution_chart ---
def create_outlier_distribution_chart(
    all_metrics_outliers_summary: DataFrame[AllMetricsOutlierSchema],
    severe_outlier_color: str,
    moderate_outlier_color: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_metrics_outliers_summary[AllMetricsOutlierSchema.metric_name],
            y=all_metrics_outliers_summary[AllMetricsOutlierSchema.total_severe_outliers],
            name="Severe outliers",
            marker_color=severe_outlier_color,
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_metrics_outliers_summary[AllMetricsOutlierSchema.metric_name],
            y=all_metrics_outliers_summary[AllMetricsOutlierSchema.total_moderate_outliers],
            name="Moderate outliers",
            marker_color=moderate_outlier_color,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0),
        title_text="Outliers",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

# --- from deepchecks__deepchecks::deepchecks/core/check_utils/multivariate_drift_utils.py::display_dist ---
def display_dist(
        train_column: pd.Series,
        test_column: pd.Series,
        fi: pd.Series,
        cat_features: Container[str],
        max_num_categories: int,
        show_categories_by: str,
        dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES
):
    """Create a distribution comparison plot for the given columns."""
    column_name = train_column.name or ''
    column_fi = fi.loc[column_name]
    title = f'Feature: {column_name} - Explains {format_percent(column_fi)} of dataset difference'

    dist_traces, xaxis_layout, yaxis_layout = feature_distribution_traces(
        train_column.dropna(),
        test_column.dropna(),
        column_name,
        is_categorical=column_name in cat_features,
        max_num_categories=max_num_categories,
        show_categories_by=show_categories_by,
        dataset_names=dataset_names
    )

    fig = go.Figure()
    fig.add_traces(dist_traces)

    return fig.update_layout(go.Layout(
        title=title,
        xaxis=xaxis_layout,
        yaxis=yaxis_layout,
        legend=dict(
            title='Dataset',
            yanchor='top',
            y=0.9,
            xanchor='left'),
        height=300
    ))

# --- from microsoft__RD-Agent::rdagent/log/ui/utils.py::curve_figure ---
def curve_figure(scores: pd.DataFrame) -> go.Figure:
    """
    scores.columns.name is the metric name, e.g., "accuracy", "f1", etc.
    scores.index is the loop index, e.g., ["L1", "L2", "L3", ...]
    scores["test"] is the test score, other columns are valid scores for different loops.
    The "ensemble" column is the ensemble score.
    The "Test scores" and "ensemble" lines are visible, while other valid scores are hidden by default.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores.index,
            y=scores["test"],
            mode="lines+markers",
            name="Test scores",
            marker=dict(symbol="diamond"),
            line=dict(shape="linear", dash="dash"),
        )
    )
    for column in scores.columns:
        if column != "test":
            fig.add_trace(
                go.Scatter(
                    x=scores.index,
                    y=scores[column],
                    mode="lines+markers",
                    name=f"{column}",
                    visible=("legendonly" if column != "ensemble" else None),
                )
            )
    fig.update_layout(title=f"Test and Valid scores (metric: {scores.columns.name})")

    return fig
