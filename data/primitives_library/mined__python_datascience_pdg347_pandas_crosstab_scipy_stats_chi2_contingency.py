# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg347::pandas.crosstab+scipy.stats.chi2_contingency
# name: pandas_scipy_primitive
# summary: Uses pandas.crosstab, scipy.stats.chi2_contingency across 2 repos
# anchor_symbols: ['pandas.crosstab', 'scipy.stats.chi2_contingency']
# observed in 2 repos: ['ModelOriented__DALEX', 'lux-org__lux']...

# --- from ModelOriented__DALEX::python/dalex/dalex/aspect/utils.py::calculate_cat_assoc_matrix ---
def calculate_cat_assoc_matrix(data, categorical_variables, n):
    from scipy.stats import chi2_contingency
    cat_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=categorical_variables, dtype=float
    )
    for i, variable_name_a in enumerate(categorical_variables):
        for j, variable_name_b in enumerate(categorical_variables):
            if i > j:
                continue   # matrix is symmetric
            elif i == j:
                cramers_V = 1
            else:
                # calculate Cramér’s V with bias correction
                cont_tab = pd.crosstab(data[variable_name_a], data[variable_name_b])
                r = cont_tab.shape[0]
                c = cont_tab.shape[1]
                chi2_test = chi2_contingency(cont_tab)
                phi = max(0, (chi2_test[0] / n) - (((r - 1) * (c - 1)) / (n - 1)))
                r = r - ((r - 1) ** 2 / (n - 1))
                c = c - ((c - 1) ** 2 / (n - 1))
                cramers_V = np.sqrt(phi / (min(c, r) - 1))
                cramers_V = checks.check_assoc_value(cramers_V)
            cat_assoc_matrix.loc[variable_name_a, variable_name_b] = cramers_V
            cat_assoc_matrix.loc[variable_name_b, variable_name_a] = cramers_V
    return cat_assoc_matrix

# --- from lux-org__lux::lux/interestingness/interestingness.py::monotonicity ---
def monotonicity(vis: Vis, attr_specs: list, ignore_identity: bool = True) -> int:
    """
    Monotonicity measures there is a monotonic trend in the scatterplot, whether linear or not.
    This score is computed as the Pearson's correlation on the ranks of x and y.
    See "Graph-Theoretic Scagnostics", Wilkinson et al 2005: https://research.tableau.com/sites/default/files/Wilkinson_Infovis-05.pdf
    Parameters
    ----------
    vis : Vis
    attr_spec: list
            List of attribute Clause objects

    ignore_identity: bool
            Boolean flag to ignore items with the same x and y attribute (score as -1)

    Returns
    -------
    int
            Score describing the strength of monotonic relationship in vis
    """
    from scipy.stats import pearsonr

    msr1 = attr_specs[0].attribute
    msr2 = attr_specs[1].attribute

    if ignore_identity and msr1 == msr2:  # remove if measures are the same
        return -1
    vxy = vis.data.dropna()
    v_x = vxy[msr1]
    v_y = vxy[msr2]

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            score = np.abs(pearsonr(v_x, v_y)[0])
        except:
            # RuntimeWarning: invalid value encountered in true_divide (occurs when v_x and v_y are uniform, stdev in denominator is zero, leading to spearman's correlation as nan), ignore these cases.
            score = -1

    if pd.isnull(score):
        return -1
    else:
        return score

# --- from lux-org__lux::lux/interestingness/interestingness.py::interestingness ---
def interestingness(vis: Vis, ldf: LuxDataFrame) -> int:
    """
    Compute the interestingness score of the vis.
    The interestingness metric is dependent on the vis type.

    Parameters
    ----------
    vis : Vis
    ldf : LuxDataFrame

    Returns
    -------
    int
            Interestingness Score
    """
    if vis.data is None or len(vis.data) == 0:
        return -1
        # raise Exception("Vis.data needs to be populated before interestingness can be computed. Run Executor.execute(vis,ldf).")
    try:
        filter_specs = utils.get_filter_specs(vis._inferred_intent)
        vis_attrs_specs = utils.get_attrs_specs(vis._inferred_intent)
        n_dim = vis._ndim
        n_msr = vis._nmsr
        n_filter = len(filter_specs)
        attr_specs = [clause for clause in vis_attrs_specs if clause.attribute != "Record"]
        dimension_lst = vis.get_attr_by_data_model("dimension")
        measure_lst = vis.get_attr_by_data_model("measure")
        v_size = len(vis.data)

        if (
            n_dim == 1
            and (n_msr == 0 or n_msr == 1)
            and ldf.current_vis is not None
            and vis.get_attr_by_channel("y")[0].data_type == "quantitative"
            and len(ldf.current_vis) == 1
            and ldf.current_vis[0].mark == "line"
            and len(get_filter_specs(ldf.intent)) > 0
        ):
            query_vc = VisList(ldf.current_vis, ldf)
            query_vis = query_vc[0]
            preprocess(query_vis)
            preprocess(vis)
            return 1 - euclidean_dist(query_vis, vis)

        # Line/Bar Chart
        if n_dim == 1 and (n_msr == 0 or n_msr == 1):
            if v_size < 2:
                return -1

            if vis.mark == "geographical":
                return n_distinct(vis, dimension_lst, measure_lst)
            if n_filter == 0:
                return unevenness(vis, ldf, measure_lst, dimension_lst)
            elif n_filter == 1:
                return deviation_from_overall(vis, ldf, filter_specs, measure_lst[0].attribute)
        # Histogram
        elif n_dim == 0 and n_msr == 1:
            if v_size < 2:
                return -1
            if n_filter == 0 and "Number of Records" in vis.data:
                if "Number of Records" in vis.data:
                    v = vis.data["Number of Records"]
                    return skewness(v)
            elif n_filter == 1 and "Number of Records" in vis.data:
                return deviation_from_overall(vis, ldf, filter_specs, "Number of Records")
            return -1
        # Scatter Plot
        elif n_dim == 0 and n_msr == 2:
            if v_size < 10:
                return -1
            if vis.mark == "heatmap":
                return weighted_correlation(
                    vis.data["xBinStart"], vis.data["yBinStart"], vis.data["count"]
                )
            if n_filter == 1:
                v_filter_size = get_filtered_size(filter_specs, vis.data)
                sig = v_filter_size / v_size
            else:
                sig = 1
            return sig * monotonicity(vis, attr_specs)
        # Scatterplot colored by Dimension
        elif n_dim == 1 and n_msr == 2:
            if v_size < 10:
                return -1
            color_attr = vis.get_attr_by_channel("color")[0].attribute

            C = ldf.cardinality[color_attr]
            if C < 40:
                return 1 / C
            else:
                return -1
        # Scatterplot colored by dimension
        elif n_dim == 1 and n_msr == 2:
            return 0.2
        # Scatterplot colored by measure
        elif n_msr == 3:
            return 0.1
        # colored line and barchart cases
        elif vis.mark == "line" and n_dim == 2:
            return 0.15
        # for colored bar chart, scoring based on Chi-square test for independence score.
        # gives higher scores to colored bar charts with fewer total categories as these charts are easier to read and thus more useful for users
        elif vis.mark == "bar" and n_dim == 2:
            from scipy.stats import chi2_contingency

            measure_column = vis.get_attr_by_data_model("measure")[0].attribute
            dimension_columns = vis.get_attr_by_data_model("dimension")

            groupby_column = dimension_columns[0].attribute
            color_column = dimension_columns[1].attribute

            contingency_tbl = pd.crosstab(
                vis.data[groupby_column],
                vis.data[color_column],
                values=vis.data[measure_column],
                aggfunc=sum,
            )

            try:
                color_cardinality = ldf.cardinality[color_column]
                groupby_cardinality = ldf.cardinality[groupby_column]
                # scale down score based on number of categories
                score = chi2_contingency(contingency_tbl)[0] * 0.9 ** (
                    color_cardinality + groupby_cardinality
                )
            except (ValueError, KeyError):
                # ValueError results if an entire column of the contingency table is 0, can happen if an applied filter results in a category having no counts
                score = -1
            return score
        # Default
        else:
            return -1
    except:
        if lux.config.interestingness_fallback:
            # Supress interestingness related issues
            warnings.warn(f"An error occurred when computing interestingness for: {vis}")
            return -1
        else:
            raise
