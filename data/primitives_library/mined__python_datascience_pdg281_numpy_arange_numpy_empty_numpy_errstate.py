# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg281::numpy.arange+numpy.empty+numpy.errstate
# name: numpy_primitive
# summary: Uses numpy.arange, numpy.empty, numpy.errstate, numpy.maximum across 2 repos
# anchor_symbols: ['numpy.arange', 'numpy.empty', 'numpy.errstate', 'numpy.maximum', 'numpy.sqrt', 'numpy.where']
# observed in 2 repos: ['rasbt__mlxtend', 'yzhao062__pyod']...

# --- from rasbt__mlxtend::mlxtend/frequent_patterns/association_rules.py::association_rules.zhangs_metric_helper ---
def zhangs_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignoring the divide by 0 warning since it is addressed in the below np.where
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)

        return zhangs_metric

# --- from yzhao062__pyod::pyod/models/ts_matrix_profile.py::_update_mp ---
def _update_mp(mp, QT, mu, sigma, const_mask, m, j, exclusion_zone, n_subseq):
    """Update the Matrix Profile for column j.

    Converts QT values to z-normalized distances and updates the
    profile wherever the new distance is smaller.

    Parameters
    ----------
    mp : np.ndarray, modified in-place
    QT : np.ndarray of shape (n_subseq,)
    mu : np.ndarray of shape (n_subseq,)
    sigma : np.ndarray of shape (n_subseq,)
    const_mask : np.ndarray of bool
    m : int
    j : int, current column index
    exclusion_zone : int
    n_subseq : int
    """
    # Compute z-normalized distance:
    #   d = sqrt(2*m*(1 - (QT - m*mu_i*mu_j) / (m*sigma_i*sigma_j)))
    # where i ranges over all subsequences

    # Denominator: m * sigma_i * sigma_j
    denom = m * sigma * sigma[j]

    # Numerator inside the (1 - ...) term
    numerator = QT - m * mu * mu[j]

    # Compute the argument of sqrt
    # Avoid division by zero: where denom is 0 (constant subsequences),
    # distance is inf
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0, numerator / denom, 0.0)

    dist_sq = 2 * m * (1 - corr)

    # Clip for numerical stability
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)

    # Set distance to inf for constant subsequences
    dist[const_mask] = np.inf
    if const_mask[j]:
        dist[:] = np.inf

    # Apply exclusion zone: ignore indices where |i - j| <= exclusion_zone
    ez_start = max(0, j - exclusion_zone)
    ez_end = min(n_subseq, j + exclusion_zone + 1)
    dist[ez_start:ez_end] = np.inf

    # Update Matrix Profile where distance is smaller
    mask = dist < mp
    mp[mask] = dist[mask]

# --- from yzhao062__pyod::pyod/models/ts_matrix_profile.py::_compute_matrix_profile ---
def _compute_matrix_profile(T, m):
    """Compute the Matrix Profile of a 1-D time series using STOMP.

    Parameters
    ----------
    T : np.ndarray of shape (n,)
        Input time series (single channel).
    m : int
        Subsequence (window) length.

    Returns
    -------
    mp : np.ndarray of shape (n - m + 1,)
        Matrix Profile values (nearest-neighbor z-normalized distances).
    """
    n = len(T)
    n_subseq = n - m + 1
    exclusion_zone = m // 4

    # --- Precompute rolling mean and std using cumulative sums ---
    cumsum = np.cumsum(T)
    cumsum2 = np.cumsum(T ** 2)

    # sum of T[i:i+m] for each subsequence i
    subseq_sum = np.empty(n_subseq)
    subseq_sum[0] = cumsum[m - 1]
    subseq_sum[1:] = cumsum[m:] - cumsum[:n - m]

    subseq_sum2 = np.empty(n_subseq)
    subseq_sum2[0] = cumsum2[m - 1]
    subseq_sum2[1:] = cumsum2[m:] - cumsum2[:n - m]

    mu = subseq_sum / m
    sigma_sq = subseq_sum2 / m - mu ** 2
    sigma_sq = np.maximum(sigma_sq, 0.0)  # numerical stability
    sigma = np.sqrt(sigma_sq)

    # Mask for constant subsequences (std < 1e-10)
    const_mask = sigma < 1e-10

    # Initialize Matrix Profile with infinity
    mp = np.full(n_subseq, np.inf)

    # --- First column (j=0): compute distance profile using MASS (FFT) ---
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2

    T_fft = np.fft.rfft(T, n=fft_size)

    # First query subsequence (reversed, then padded)
    query = T[:m][::-1]
    Q_fft = np.fft.rfft(query, n=fft_size)

    # QT[i] = dot product of T[i:i+m] and T[0:m]
    QT_full = np.fft.irfft(T_fft * Q_fft, n=fft_size)
    QT = QT_full[m - 1:m - 1 + n_subseq].copy()

    # Compute distance for j=0
    _update_mp(mp, QT, mu, sigma, const_mask, m, 0, exclusion_zone, n_subseq)

    # Keep a copy of QT for incremental updates
    QT_prev = QT.copy()

    # --- STOMP: incremental updates for j=1..n_subseq-1 ---
    for j in range(1, n_subseq):
        # Incremental QT update:
        # QT_new[i] = QT_old[i-1] - T[i-1]*T[j-1] + T[i+m-1]*T[j+m-1]
        QT_new = np.empty(n_subseq)

        # QT_new[0] must be computed as a direct dot product
        QT_new[0] = np.dot(T[:m], T[j:j + m])

        # Vectorized incremental update for i=1..n_subseq-1
        i_indices = np.arange(1, n_subseq)
        QT_new[1:] = (QT_prev[:-1]
                       - T[i_indices - 1] * T[j - 1]
                       + T[i_indices + m - 1] * T[j + m - 1])

        _update_mp(mp, QT_new, mu, sigma, const_mask, m, j,
                   exclusion_zone, n_subseq)

        QT_prev = QT_new

    return mp

# --- from rasbt__mlxtend::mlxtend/frequent_patterns/association_rules.py::association_rules ---
def association_rules(
    df: pd.DataFrame,
    num_itemsets: Optional[int] = 1,
    df_orig: Optional[pd.DataFrame] = None,
    null_values=False,
    metric="confidence",
    min_threshold=0.8,
    support_only=False,
    return_metrics: list = _metrics,
) -> pd.DataFrame:
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']

    df_orig : pandas DataFrame (default: None)
      DataFrame with original input data. Only provided when null_values exist

    num_itemsets : int (default: 1)
      Number of transactions in original input data (df_orig)

    null_values : bool (default: False)
      In case there are null values as NaNs in the original input data

    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      **Automatically set to 'support' if `support_only=True`.**
      Otherwise, supported metrics are 'support', 'confidence', 'lift',
      'leverage', 'conviction' and 'zhangs_metric'
      These metrics are computed as follows:

      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]\n
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n
      - leverage(A->C) = support(A->C) - support(A)*support(C),
        range: [-1, 1]\n
      - conviction = [1 - support(C)] / [1 - confidence(A->C)],
        range: [0, inf]\n
      - zhangs_metric(A->C) =
        leverage(A->C) / max(support(A->C)*(1-support(A)), support(A)*(support(C)-support(A->C)))
        range: [-1,1]\n

    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric,
      via the `metric` parameter,
      to decide whether a candidate rule is of interest.

    support_only : bool (default: False)
      Only computes the rule support and fills the other
      metric columns with NaNs. This is useful if:

      a) the input DataFrame is incomplete, e.g., does
      not contain support values for all rule antecedents
      and consequents

      b) you simply want to speed up the computation because
      you don't need the other metrics.

    Returns
    ----------
    pandas DataFrame with columns "antecedents" and "consequents"
      that store itemsets, plus the scoring metric columns:
      "antecedent support", "consequent support",
      "support", "confidence", "lift",
      "leverage", "conviction"
      of all rules for which
      metric(rule) >= min_threshold.
      Each entry in the "antecedents" and "consequents" columns are
      of type `frozenset`, which is a Python built-in type that
      behaves similarly to sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

    """
    # if null values exist, df_orig must be provided
    if null_values and df_orig is None:
        raise TypeError("If null values exist, df_orig must be provided.")

    # if null values exist, num_itemsets must be provided
    if null_values and num_itemsets == 1:
        raise TypeError("If null values exist, num_itemsets must be provided.")

    # check for valid input
    fpc.valid_input_check(df_orig, null_values)

    if not df.shape[0]:
        raise ValueError(
            "The input DataFrame `df` containing " "the frequent itemsets is empty."
        )

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    def kulczynski_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        conf_AC = sAC * (num_itemsets - disAC) / (sA * (num_itemsets - disA) - dis_int)
        conf_CA = sAC * (num_itemsets - disAC) / (sC * (num_itemsets - disC) - dis_int_)
        kulczynski = (conf_AC + conf_CA) / 2
        return kulczynski

    def conviction_helper(conf, sC):
        conviction = np.empty(conf.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            conf = conf[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[conf < 1.0] = (1.0 - sC[conf < 1.0]) / (1.0 - conf[conf < 1.0])

        return conviction

    def zhangs_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignoring the divide by 0 warning since it is addressed in the below np.where
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)

        return zhangs_metric

    def jaccard_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        numerator = metric_dict["support"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )
        denominator = sA + sC - numerator

        jaccard_metric = numerator / denominator
        return jaccard_metric

    def certainty_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        certainty_num = (
            metric_dict["confidence"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
            - sC
        )
        certainty_denom = 1 - sC

        cert_metric = np.where(certainty_denom == 0, 0, certainty_num / certainty_denom)
        return cert_metric

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, ___, ____, _____, ______, _______, ________: sA,
        "consequent support": lambda _, __, sC, ____, _____, ______, _______, ________: sC,
        "support": lambda sAC, _, __, ___, ____, _____, ______, _______: sAC,
        "confidence": lambda sAC, sA, _, disAC, disA, __, dis_int, ___: (
            sAC * (num_itemsets - disAC)
        )
        / (sA * (num_itemsets - disA) - dis_int),
        "lift": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: metric_dict[
            "confidence"
        ](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
        / sC,
        "representativity": lambda _, __, ___, disAC, ____, ______, _______, ________: (
            num_itemsets - disAC
        )
        / num_itemsets,
        "leverage": lambda sAC, sA, sC, _, __, ____, _____, ______: metric_dict[
            "support"
        ](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
        - sA * sC,
        "conviction": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: conviction_helper(
            metric_dict["confidence"](
                sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
            ),
            sC,
        ),
        "zhangs_metric": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: zhangs_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "jaccard": lambda sAC, sA, sC, _, __, ____, _____, ______: jaccard_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "certainty": lambda sAC, sA, sC, _, __, ____, _____, ______: certainty_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "kulczynski": lambda sAC, sA, sC, _, __, ____, _____, ______: kulczynski_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
    }

    # check for metric compliance
    if support_only:
        metric = "support"
    else:
        if metric not in metric_dict.keys():
            raise ValueError(
                "Metric must be 'confidence' or 'lift', got '{}'".format(metric)
            )

    # get dict of {frequent itemset} -> support
    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(
        lambda x: frozenset(
            int(item) if isinstance(item, np.generic) else item for item in x
        )
    )
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # Define the disabled df, assign columns from original df to be the same on the disabled.
    if null_values:
        first_itemset = next(iter(frequent_items_dict.keys()))
        df_orig = df_orig.copy()
        disabled = df_orig.copy()
        disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
            (disabled == 0) | (disabled == 1), np.nan, 0
        )
        disabled = pd.DataFrame(disabled)
        if all(isinstance(key, str) for key in first_itemset):
            disabled.columns = df_orig.columns

        if all(isinstance(key, (np.integer, int)) for key in first_itemset):
            cols = np.arange(0, len(df_orig.columns), 1)
            disabled.columns = cols
            df_orig = df_orig.rename(columns=dict(zip(df_orig.columns, cols)))

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k) - 1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    # support doesn't need these,
                    # hence, placeholders should suffice
                    sA = None
                    sC = None
                    disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0

                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]

                        # if the input dataframe is complete
                        if not null_values:
                            disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0

                        else:
                            an = list(antecedent)
                            con = list(consequent)
                            an.extend(con)

                            # select data of antecedent, consequent and combined from disabled
                            dec = disabled.loc[:, an]
                            _dec = disabled.loc[:, list(antecedent)]
                            __dec = disabled.loc[:, list(consequent)]

                            # select data of antecedent and consequent from original
                            dec_ = df_orig.loc[:, list(antecedent)]
                            dec__ = df_orig.loc[:, list(consequent)]

                            # disabled counts
                            disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0
                            for i in range(len(dec.index)):
                                # select the i-th iset from the disabled dataset
                                item_comb = list(dec.iloc[i, :])
                                item_dis_an = list(_dec.iloc[i, :])
                                item_dis_con = list(__dec.iloc[i, :])

                                # select the i-th iset from the original dataset
                                item_or_an = list(dec_.iloc[i, :])
                                item_or_con = list(dec__.iloc[i, :])

                                # check and keep count if there is a null value in combined, antecedent, consequent
                                if 1 in set(item_comb):
                                    disAC += 1
                                if 1 in set(item_dis_an):
                                    disA += 1
                                if 1 in item_dis_con:
                                    disC += 1

                                # check and keep count if there is a null value in consequent AND all items are present in antecedent
                                if (1 in item_dis_con) and all(
                                    j == 1 for j in item_or_an
                                ):
                                    dis_int += 1

                                # check and keep count if there is a null value in antecedent AND all items are present in consequent
                                if (1 in item_dis_an) and all(
                                    j == 1 for j in item_or_con
                                ):
                                    dis_int_ += 1

                    except KeyError as e:
                        s = (
                            str(e) + "You are likely getting this error"
                            " because the DataFrame is missing "
                            " antecedent and/or consequent "
                            " information."
                            " You can try using the "
                            " `support_only=True` option"
                        )
                        raise KeyError(s)
                    # check for the threshold

                score = metric_dict[metric](
                    sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
                )
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append(
                        [sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_]
                    )

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        if support_only:
            sAC = rule_supports[0]
            for m in return_metrics:
                df_res[m] = np.nan
            df_res["support"] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            disAC = rule_supports[3]
            disA = rule_supports[4]
            disC = rule_supports[5]
            dis_int = rule_supports[6]
            dis_int_ = rule_supports[7]

            for m in return_metrics:
                df_res[m] = metric_dict[m](
                    sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
                )

        return df_res
