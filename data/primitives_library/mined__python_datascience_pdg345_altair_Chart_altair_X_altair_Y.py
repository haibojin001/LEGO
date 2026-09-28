# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg345::altair.Chart+altair.X+altair.Y
# name: altair_primitive
# summary: Uses altair.Chart, altair.X, altair.Y across 4 repos
# anchor_symbols: ['altair.Chart', 'altair.X', 'altair.Y']
# observed in 4 repos: ['encord-team__encord-active', 'lux-org__lux', 'ploomber__sklearn-evaluation', 'stitchfix__hamilton']...

# --- from encord-team__encord-active::src/encord_active/lib/charts/partition_histogram.py::get_partition_histogram ---
def get_partition_histogram(balanced_df: pd.DataFrame, metric: str):
    return (
        alt.Chart(balanced_df)
        .mark_bar(
            binSpacing=0,
        )
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=50)),
            y="count()",
            color="partition:N",
            tooltip=["partition", "count()"],
        )
    )

# --- from lux-org__lux::lux/vislib/altair/Histogram.py::compute_bin_width ---
def compute_bin_width(series):
    """
    Helper function that returns optimal bin size via Freedman Diaconis's Rule
    Source: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """
    import numpy as np

    data = np.asarray(series)
    num_pts = data.size
    IQR = np.subtract(*np.percentile(data, [75, 25]))
    size = 2 * IQR * (num_pts**-1 / 3)
    return round(size * 3.5, 2)

# --- from encord-team__encord-active::src/encord_active/lib/charts/performance_by_metric.py::performance_average_rule ---
def performance_average_rule(
    indicator_mean: float, scope: PredictionMatchScope, color_params: Optional[dict] = None
) -> alt_api.Chart:
    title = CHART_TITLES[scope]
    title_shorthand = "".join(w[0].upper() for w in title.split())
    color_params = color_params or {}
    return (
        alt.Chart(pd.DataFrame({"y": [indicator_mean], "average": ["Average"]}))
        .mark_rule()
        .encode(
            alt.Y("y"),
            alt.Color("average:N", **color_params),
            strokeDash=alt.value([5, 5]),
            tooltip=[alt.Tooltip("y", title=f"Average {title_shorthand}", format=FLOAT_FMT)],
        )
    )

# --- from stitchfix__hamilton::tests/integrations/pandera/test_pandera_data_quality.py::test_basic_pandera_decorator_dataframe_passes ---
def test_basic_pandera_decorator_dataframe_passes():
    schema = pa.DataFrameSchema(
        {
            "column1": pa.Column(int),
            "column2": pa.Column(float, pa.Check(lambda s: s < -1.2)),
            # you can provide a list of validators
            "column3": pa.Column(
                str,
                [
                    pa.Check(lambda s: s.str.startswith("value")),
                    pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2),
                ],
            ),
        },
        index=pa.Index(int),
        strict=True,
    )

    df = pd.DataFrame(
        {
            "column1": [5, 1, 0],
            "column2": [-2.0, -1.9, -20.0],
            "column3": ["value_0", "value_1", "value_2"],
        }
    )
    validator = pandera_validators.PanderaDataFrameValidator(schema=schema, importance="warn")
    validation_result = validator.validate(df)
    assert validation_result.passes

# --- from stitchfix__hamilton::tests/integrations/pandera/test_pandera_data_quality.py::test_basic_pandera_decorator_dataframe_fails ---
def test_basic_pandera_decorator_dataframe_fails():
    schema = pa.DataFrameSchema(
        {
            "column1": pa.Column(int),
            "column2": pa.Column(float, pa.Check(lambda s: s < -1.2)),
            # you can provide a list of validators
            "column3": pa.Column(
                str,
                [
                    pa.Check(lambda s: s.str.startswith("value")),
                    pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2),
                ],
            ),
        },
        index=pa.Index(int),
        strict=True,
    )

    df = pd.DataFrame({"column1": [5, 1, np.nan]})
    validator = pandera_validators.PanderaDataFrameValidator(schema=schema, importance="warn")
    validation_result = validator.validate(df)
    assert not validation_result.passes
    assert (
        "A total of 4 schema errors were found" in validation_result.message
    )  # TODO -- ensure this will stay constant with the contract

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/plot/confusion_matrix_interactive.py::_plot_cm_chart ---
def _plot_cm_chart(df, selection, alt):
    if selection is not None:
        color = alt.condition(
            selection,
            "confusion_matrix:N",
            alt.value("lightgray"),
            scale=alt.Scale(scheme="oranges"),
        )
    else:
        color = alt.Color(scale=alt.Scale(scheme="oranges"))

    heatmap = (
        alt.Chart(df, title="Confusion Matrix")
        .mark_rect()
        .encode(
            alt.X("predicted:N"),
            alt.Y("actual:N"),
            color=color,
        )
    )

    text = (
        alt.Chart(df, title="Confusion Matrix")
        .mark_text(baseline="middle", fontSize=25, fontWeight="bold")
        .encode(
            x="predicted:N",
            y="actual:N",
            text="confusion_matrix:N",
            color=alt.condition(
                alt.datum.confusion_matrix > 0,
                alt.value("black"),
                alt.value("black"),
            ),
        )
    )
    cm_chart = (heatmap + text).properties(width=600, height=480)
    return cm_chart

# --- from lux-org__lux::lux/vislib/altair/ScatterChart.py::ScatterChart.initialize_chart ---
def initialize_chart(self):
        x_attr = self.vis.get_attr_by_channel("x")[0]
        y_attr = self.vis.get_attr_by_channel("y")[0]

        x_attr_abv = str(x_attr.attribute)
        y_attr_abv = str(y_attr.attribute)

        if len(x_attr_abv) > 25:
            x_attr_abv = x_attr.attribute[:15] + "..." + x_attr.attribute[-10:]
        if len(y_attr_abv) > 25:
            y_attr_abv = y_attr.attribute[:15] + "..." + y_attr.attribute[-10:]

        x_min = self.vis.min_max[x_attr.attribute][0]
        x_max = self.vis.min_max[x_attr.attribute][1]

        y_min = self.vis.min_max[y_attr.attribute][0]
        y_max = self.vis.min_max[y_attr.attribute][1]

        if isinstance(x_attr.attribute, str):
            x_attr.attribute = x_attr.attribute.replace(".", "")
        if isinstance(y_attr.attribute, str):
            y_attr.attribute = y_attr.attribute.replace(".", "")
        self.data = AltairChart.sanitize_dataframe(self.data)
        chart = (
            alt.Chart(self.data)
            .mark_circle()
            .encode(
                x=alt.X(
                    str(x_attr.attribute),
                    scale=alt.Scale(domain=(x_min, x_max)),
                    type=x_attr.data_type,
                    axis=alt.Axis(title=x_attr_abv),
                ),
                y=alt.Y(
                    str(y_attr.attribute),
                    scale=alt.Scale(domain=(y_min, y_max)),
                    type=y_attr.data_type,
                    axis=alt.Axis(title=y_attr_abv),
                ),
            )
        )
        # Setting tooltip as non-null
        chart = chart.configure_mark(tooltip=alt.TooltipContent("encoding"))
        chart = chart.interactive()  # Enable Zooming and Panning

        #####################################
        ## Constructing Altair Code String ##
        #####################################

        self.code += "import altair as alt\n"
        dfname = "placeholder_variable"
        self.code += f"""
		chart = alt.Chart({dfname}).mark_circle().encode(
		    x=alt.X('{x_attr.attribute}',scale=alt.Scale(domain=({x_min}, {x_max})),type='{x_attr.data_type}', axis=alt.Axis(title='{x_attr_abv}')),
		    y=alt.Y('{y_attr.attribute}',scale=alt.Scale(domain=({y_min}, {y_max})),type='{y_attr.data_type}', axis=alt.Axis(title='{y_attr_abv}'))
		)
		chart = chart.configure_mark(tooltip=alt.TooltipContent('encoding')) # Setting tooltip as non-null
		chart = chart.interactive() # Enable Zooming and Panning
		"""
        return chart
