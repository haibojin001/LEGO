# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg534::bokeh.layouts.row+bokeh.models.ColumnDataSource+bokeh.models.HoverTool
# name: bokeh_primitive
# summary: Uses bokeh.layouts.row, bokeh.models.ColumnDataSource, bokeh.models.HoverTool, bokeh.models.Panel across 2 repos
# anchor_symbols: ['bokeh.layouts.row', 'bokeh.models.ColumnDataSource', 'bokeh.models.HoverTool', 'bokeh.models.Panel', 'bokeh.plotting.figure']
# observed in 2 repos: ['MentatInnovations__datastream.io', 'sfu-db__dataprep']...

# --- from sfu-db__dataprep::dataprep/eda/distribution/render.py::line_viz ---
def line_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    ttl_grps: int,
) -> Panel:
    """
    Render multi-line chart
    """
    # pylint: disable=too-many-arguments,too-many-locals
    palette = CATEGORY20 * (len(df) // len(CATEGORY20) + 1)
    title = _make_title({f"{x}_ttl": ttl_grps, f"{x}_shw": len(df)}, x, y)
    df.index = df.index.astype(str)

    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=title,
        toolbar_location=None,
        tools=[],
        y_axis_type=yscale,
    )

    # bin endpoints for all histograms
    bins = df[0].iloc[0][1]
    # plot the value for a histgram bin at its midpoint
    ticks = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    # format the bin intervals
    intvls = _format_bin_intervals(bins)

    lns: Dict[str, Figure] = {}
    # add the lines
    for grp, (cnts, _), color in zip(df.index, df[0], palette):
        grp_name = (grp[:14] + "...") if len(grp) > 15 else grp
        source = ColumnDataSource({"x": ticks, "y": cnts, "intvls": intvls})
        lns[grp_name] = fig.line(x="x", y="y", color=color, source=source)
        tooltips = [(f"{x}", f"{grp}"), ("Frequency", "@y"), (f"{y} bin", "@intvls")]
        fig.add_tools(HoverTool(renderers=[lns[grp_name]], tooltips=tooltips))

    fig.add_layout(Legend(items=[(x, [lns[x]]) for x in lns]), "left")
    tweak_figure(fig)
    fig.yaxis.axis_label = "Frequency"
    fig.xaxis.axis_label = y
    _format_axis(fig, bins[0], bins[-1], "x")
    if yscale == "linear":
        yvals = [val for cnts, _ in df[0] for val in cnts]
        _format_axis(fig, min(yvals), max(yvals), "y")

    return Panel(child=row(fig), title="Line Chart")

# --- from sfu-db__dataprep::dataprep/eda/distribution/render.py::heatmap_viz ---
def heatmap_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    grp_cnt_stats: Dict[str, int],
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render a heatmap
    """
    # pylint: disable=too-many-arguments
    title = _make_title(grp_cnt_stats, x, y)

    source = ColumnDataSource(data=df)
    palette = RDBU[(len(RDBU) // 2 - 1) :]
    mapper = LinearColorMapper(palette=palette, low=df["cnt"].min() - 0.01, high=df["cnt"].max())
    if grp_cnt_stats[f"{x}_shw"] > 60:
        plot_width = 16 * grp_cnt_stats[f"{x}_shw"]
    if grp_cnt_stats[f"{y}_shw"] > 10:
        plot_height = 70 + 18 * grp_cnt_stats[f"{y}_shw"]
    fig = figure(
        x_range=sorted(list(set(df[x]))),
        y_range=sorted(list(set(df[y]))),
        toolbar_location=None,
        tools=[],
        x_axis_location="below",
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    renderer = fig.rect(
        x=x,
        y=y,
        width=1,
        height=1,
        source=source,
        line_color=None,
        fill_color=transform("cnt", mapper),
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=7),
        formatter=PrintfTickFormatter(format="%d"),
    )
    fig.add_tools(
        HoverTool(
            tooltips=[
                (x, f"@{{{x}}}"),
                (y, f"@{{{y}}}"),
                ("Count", "@cnt"),
            ],
            mode="mouse",
            renderers=[renderer],
        )
    )
    fig.add_layout(color_bar, "right")

    tweak_figure(fig, "heatmap")
    fig.yaxis.formatter = FuncTickFormatter(
        code="""
            if (tick.length > 15) return tick.substring(0, 14) + '...';
            else return tick;
        """
    )
    return Panel(child=fig, title="Heat Map")

# --- from MentatInnovations__datastream.io::dsio/dashboard/bokeh.py::generate_dashboard.make_document ---
def make_document(doc):
        """ Generates the dashboard document """
        # Initialize the data source
        data = {'time': []}
        for sensor in sensors:
            sensor_score = 'SCORE_%s' % sensor
            sensor_flag = 'FLAG_%s' % sensor
            data[sensor] = []
            data[sensor_score] = []
            data[sensor_flag] = []
        source = ColumnDataSource(data=data)

        # Add figure for each sensor
        tools = 'pan,wheel_zoom,xbox_select,reset'
        figures = []
        for sensor in sensors:
            fig = figure(title=sensor, tools=tools, x_axis_type='datetime',
                         plot_width=600, plot_height=300)
            fig.line('time', sensor, source=source)
            sensor_score = 'SCORE_%s' % sensor
            sensor_flag = 'FLAG_%s' % sensor
            fig.circle('time', sensor, size=5, source=source, color='red', fill_alpha=sensor_flag, line_alpha=0)
            hover = HoverTool(
                tooltips=[
                    ("time", "@time{%F %T}"),
                    ("value", "@%s" % sensor),
                    ("score", "@%s" % sensor_score),
                ],
                formatters={"time": "datetime"},
                mode='vline'
            )
            fig.add_tools(hover)

            if figures: # share the x-axis across all figures
                fig.x_range = figures[0].x_range

            figures.append(fig)

        grid = gridplot(figures, ncols=cols, sizing_mode='scale_width')
        doc.title = title
        doc.add_root(grid)

        def update():
            """ Check the queue for updates sent by the restreamer thread
                and pass them over to the bokeh data soure """
            data = update_queue.get().to_dict('list')
            source.stream(data)

        if update_queue: # Update every second
            doc.add_periodic_callback(update, 1000)
