# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg532::dask.compute+dask.dataframe.concat+dask.delayed
# name: dask_primitive
# summary: Uses dask.compute, dask.dataframe.concat, dask.delayed across 2 repos
# anchor_symbols: ['dask.compute', 'dask.dataframe.concat', 'dask.delayed']
# observed in 2 repos: ['sfu-db__dataprep', 'xorbitsai__xorbits']...

# --- from xorbitsai__xorbits::python/xorbits/_mars/contrib/dask/tests/test_dask.py::test_multiple_objects ---
def test_multiple_objects(setup_cluster):
    import dask

    def inc(x: int):
        return x + 1

    test_list = [dask.delayed(inc)(i) for i in range(10)]
    test_tuple = tuple(dask.delayed(inc)(i) for i in range(10))
    test_dict = {str(i): dask.delayed(inc)(i) for i in range(10)}

    for test_obj in (test_list, test_tuple, test_dict):
        assert dask.compute(test_obj) == dask.compute(
            test_obj, scheduler=mars_scheduler
        )

# --- from sfu-db__dataprep::dataprep/eda/diff/compute/multiple_column.py::compare_multiple_col ---
def compare_multiple_col(
    df_list: List[dd.DataFrame],
    x: str,
    cfg: Config,
) -> Intermediate:
    """
    Compute function for plot_diff([df...],x)

    Parameters
    ----------
    df_list
        Dataframe sequence to be compared.
    x
        Name of the column to be compared
    cfg
        Config instance
    """
    aligned_dfs = dd.concat(df_list, axis=1)
    baseline: int = cfg.diff.baseline
    srs = Srs(aligned_dfs[x])
    data: List[Any] = []
    col_dtype = srs.self_map(detect_dtype)
    if len(col_dtype) > 1:
        col_dtype = col_dtype[baseline]
    else:
        col_dtype = col_dtype[0]

    if is_dtype(col_dtype, Continuous()):
        data.append((_cont_calcs(srs.apply("dropna"), cfg, df_list, x)))
        stats = calc_stats_cont(srs, cfg)
        stats, data = dask.compute(stats, data)

        return Intermediate(col=x, data=data, stats=stats, visual_type="comparison_continuous")
    else:
        return Intermediate()

# --- from sfu-db__dataprep::dataprep/eda/diff/compute/multiple_df.py::_cont_calcs ---
def _cont_calcs(srs: Srs, cfg: Config) -> Dict[str, List[Any]]:
    """
    Computations for a continuous column in plot_diff([df1, df2, ..., dfn])
    """

    data: Dict[str, List[Any]] = {}

    # drop infinite values
    mask = srs.apply("isin", {np.inf, -np.inf})
    srs = Srs(srs.getmask(mask, inverse=True), agg=True)
    min_max = srs.apply(
        "map_partitions", lambda x: pd.Series([x.max(), x.min()]), meta=pd.Series([], dtype=float)
    ).data
    min_max_comp = []
    if cfg.diff.density:
        for min_max_value in dask.compute(min_max)[0]:
            min_max_comp.append(math.isclose(min_max_value.min(), min_max_value.max()))
    min_max = dd.concat(min_max).repartition(npartitions=1)

    # histogram
    data["hist"] = srs.self_map(
        da.histogram, bins=cfg.hist.bins, range=(min_max.min(), min_max.max())
    )

    # compute the density histogram
    if cfg.diff.density:
        data["dens"] = srs.self_map(
            da.histogram,
            condition=min_max_comp,
            bins=cfg.kde.bins,
            range=(min_max.min(), min_max.max()),
            density=True,
        )
        # gaussian kernel density estimate
        data["kde"] = []
        sample_data = dask.compute(
            srs.apply(
                "map_partitions",
                lambda x: x.sample(min(1000, x.shape[0])),
                meta=pd.Series([], dtype=float),
            ).data
        )
        for ind in range(len(sample_data[0])):
            data["kde"].append(gaussian_kde(sample_data[0][ind]))

    return data
