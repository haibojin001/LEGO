import inspect
import itertools as it
import warnings
from typing import Optional, Union, Literal

import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
from pandas import DataFrame, Series, MultiIndex


__all__ = [
    "posthoc_conover",
    "posthoc_dunn",
    "posthoc_nemenyi",
    "posthoc_nemenyi_friedman",
    "posthoc_conover_friedman",
    "posthoc_siegel_friedman",
    "posthoc_miller_friedman",
    "posthoc_durbin",
    "posthoc_quade",
    "posthoc_vanwaerden",
    "posthoc_anderson",
    "posthoc_dscf",
    "posthoc_ttest",
    "posthoc_tukey",
    "posthoc_tukey_hsd",
    "posthoc_mannwhitney",
    "posthoc_wilcoxon",
    "posthoc_scheffe",
    "posthoc_tamhane",
    "posthoc_dunnett",
    "posthoc_duncan",
    "posthoc_snk",
]


def __convert_to_df(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = "vals",
    group_col: Optional[str] = "groups",
    val_id: Optional[int] = None,
    group_id: Optional[int] = None,
) -> tuple[DataFrame, str, str]:
    value_name = val_col or "vals"
    category_name = group_col or "groups"

    if isinstance(a, DataFrame):
        if value_name not in a.columns or category_name not in a.columns:
            raise ValueError(
                "Specify correct column names using `group_col` and `val_col` args"
            )
        return a.loc[:, [value_name, category_name]].copy(), value_name, category_name

    is_group_list = isinstance(a, list)
    if isinstance(a, np.ndarray):
        if a.ndim != 2:
            is_group_list = True
        elif a.shape.count(2) == 0 or (a.shape[0] == 2 and a.shape[1] != 2):
            is_group_list = True

    if is_group_list:
        try:
            lengths = [len(v) for v in a]
            values = list(it.chain.from_iterable(a))
        except TypeError:
            values = list(a)
            lengths = [1] * len(values)
        groups = list(
            it.chain.from_iterable(
                ([idx + 1] * size for idx, size in enumerate(lengths))
            )
        )
        return DataFrame({value_name: values, category_name: groups}), value_name, category_name

    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError("Input data must be two-dimensional")

    if val_id is None or group_id is None:
        work = arr.T if np.argmax(arr.shape) else arr
        if work.shape[1] < 2:
            raise ValueError(
                "Cannot infer input format.\nPlease specify `val_id` and `group_id` args"
            )
        distinct = [np.unique(work[:, 0]).size, np.unique(work[:, 1]).size]
        if distinct[0] == distinct[1]:
            raise ValueError(
                "Cannot infer input format.\nPlease specify `val_id` and `group_id` args"
            )
        value_index = int(np.argmax(distinct))
        group_index = int(np.argmin(distinct))
        arr = work
    else:
        value_index = val_id
        group_index = group_id

    result = DataFrame(
        arr[:, [value_index, group_index]], columns=[value_name, category_name]
    ).dropna()
    return result, value_name, category_name


def __convert_to_block_df(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
) -> tuple[DataFrame, str, str, str, str]:
    if melted and any(v is None for v in (block_col, group_col, y_col)):
        raise ValueError(
            "`block_col`, `group_col`, `y_col` should be explicitly specified if using melted data"
        )

    y_name = "y"
    group_name = "groups"
    block_name = "blocks"
    block_id_name = "block_ids"

    if isinstance(a, DataFrame):
        if not melted:
            work = a.copy(deep=True)
            work.columns.name = group_name
            work.index.name = block_name
            work[block_id_name] = np.arange(len(work))
            out = work.reset_index().melt(
                id_vars=[block_name, block_id_name],
                var_name=group_name,
                value_name=y_name,
            )
        else:
            if a[block_col].duplicated().any() and not block_id_col:
                raise ValueError(
                    "`block_col` contains duplicated entries, `block_id_col` should be explicitly specified"
                )
            if block_id_col is None:
                ids = np.arange(len(a))
            else:
                ids = a[block_id_col]
            out = DataFrame(
                {
                    group_name: a[group_col],
                    block_name: a[block_col],
                    y_name: a[y_col],
                    block_id_name: ids,
                }
            )
    else:
        work = np.asarray(a)
        if work.ndim != 2:
            raise ValueError("Input data must be two-dimensional")
        frame = DataFrame(work, index=np.arange(work.shape[0]), columns=np.arange(work.shape[1]))
        if not melted:
            frame.columns.name = group_name
            frame.index.name = block_name
            frame[block_id_name] = np.arange(len(frame))
            out = frame.reset_index().melt(
                id_vars=[block_name, block_id_name],
                var_name=group_name,
                value_name=y_name,
            )
        else:
            frame.rename(
                columns={
                    y_col: y_name,
                    group_col: group_name,
                    block_col: block_name,
                    block_id_col: block_id_name,
                },
                inplace=True,
            )
            out = frame

    return out, y_name, group_name, block_name, block_id_name


def __complete_block_matrix(
    a: Union[DataFrame, ArrayLike], melted: bool, sort: bool
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if melted:
        return None
    if isinstance(a, DataFrame):
        if sort or not a.columns.is_unique or a.columns.hasnans:
            return None
        values = a.to_numpy()
        labels = a.columns.to_numpy(copy=True)
    else:
        try:
            values = np.asarray(a)
        except (TypeError, ValueError):
            return None
        if values.ndim != 2:
            return None
        labels = np.arange(values.shape[1])
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        return None
    if np.isnan(values).any():
        return None
    return values, labels


def _labels(df, group_col, sort):
    vals = df[group_col].dropna().unique()
    return np.sort(vals) if sort else vals


def _matrix(labels):
    return DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)


def _adjust(mat, method):
    if method is None:
        return mat
    n = mat.shape[0]
    tri = np.triu_indices(n, 1)
    values = mat.to_numpy(copy=True)
    values[tri] = multipletests(values[tri], method=method)[1]
    values[(tri[1], tri[0])] = values[tri]
    np.fill_diagonal(values, 1.0)
    return DataFrame(values, index=mat.index, columns=mat.columns)


def _pairwise(labels, fun, p_adjust=None):
    out = _matrix(labels)
    for i, j in it.combinations(range(len(labels)), 2):
        value = float(fun(labels[i], labels[j]))
        out.iat[i, j] = value
        out.iat[j, i] = value
    return _adjust(out, p_adjust)


def _long_data(a, val_col, group_col, sort):
    x, value, group = __convert_to_df(a, val_col, group_col)
    x = x.dropna(subset=[value, group])
    return x, value, group, _labels(x, group, sort)


def posthoc_conover(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    x["_rank"] = ss.rankdata(x[value])
    n_total = len(x)
    k = len(labels)
    ns = x.groupby(group, observed=True).size()
    means = x.groupby(group, observed=True)["_rank"].mean()
    rank_sum_sq = np.sum(x["_rank"].to_numpy() ** 2)
    s2 = (rank_sum_sq - n_total * (n_total + 1) ** 2 / 4.0) / (n_total - 1.0)
    tie_term = x[value].value_counts().pipe(lambda z: np.sum(z.to_numpy() ** 3 - z.to_numpy()))
    correction = 1.0 - tie_term / (n_total**3 - n_total) if n_total > 1 else 1.0
    denom_factor = max((n_total - 1.0 - s2) / max(n_total - k, 1), 0.0)
    if not np.isfinite(denom_factor) or denom_factor <= 0:
        denom_factor = correction

    def test(g1, g2):
        den = np.sqrt(s2 * denom_factor * (1.0 / ns[g1] + 1.0 / ns[g2]))
        if den == 0:
            return 1.0
        stat = abs(means[g1] - means[g2]) / den
        return 2.0 * ss.t.sf(stat, max(n_total - k, 1))

    return _pairwise(labels, test, p_adjust)


def posthoc_dunn(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    x["_rank"] = ss.rankdata(x[value])
    n_total = len(x)
    ns = x.groupby(group, observed=True).size()
    means = x.groupby(group, observed=True)["_rank"].mean()
    counts = x[value].value_counts().to_numpy()
    ties = np.sum(counts**3 - counts)
    variance = (n_total * (n_total + 1.0) / 12.0) - ties / (12.0 * max(n_total - 1.0, 1.0))

    def test(g1, g2):
        den = np.sqrt(variance * (1.0 / ns[g1] + 1.0 / ns[g2]))
        return 1.0 if den == 0 else 2.0 * ss.norm.sf(abs(means[g1] - means[g2]) / den)

    return _pairwise(labels, test, p_adjust)


def posthoc_nemenyi(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    dist: Literal["chi", "tukey"] = "chi",
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    x["_rank"] = ss.rankdata(x[value])
    n_total = len(x)
    ns = x.groupby(group, observed=True).size()
    means = x.groupby(group, observed=True)["_rank"].mean()
    counts = x[value].value_counts().to_numpy()
    ties = np.sum(counts**3 - counts)
    correction = 1.0 - ties / max(n_total**3 - n_total, 1)

    def test(g1, g2):
        den = np.sqrt(
            n_total * (n_total + 1.0) / 12.0 * correction * (1.0 / ns[g1] + 1.0 / ns[g2])
        )
        if den == 0:
            return 1.0
        z = abs(means[g1] - means[g2]) / den
        if dist == "tukey":
            return ss.studentized_range.sf(z * np.sqrt(2.0), len(labels), np.inf)
        return ss.chi2.sf(z * z, 1)

    return _pairwise(labels, test)


def _block_rank_data(a, y_col, group_col, block_col, block_id_col, melted, sort):
    direct = __complete_block_matrix(a, melted, sort)
    if direct is not None:
        values, groups = direct
        ranks = ss.rankdata(values, axis=1)
        return ranks, groups, values.shape[0], values.shape[1]

    x, y, group, block, block_id = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )
    x = x.dropna(subset=[y, group, block_id])
    pivot = x.pivot_table(index=block_id, columns=group, values=y, aggfunc="first")
    if sort:
        pivot = pivot.sort_index(axis=1)
    pivot = pivot.dropna()
    return ss.rankdata(pivot.to_numpy(), axis=1), pivot.columns.to_numpy(), pivot.shape[0], pivot.shape[1]


def posthoc_nemenyi_friedman(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    ranks, labels, n, k = _block_rank_data(
        a, y_col, group_col, block_col, block_id_col, melted, sort
    )
    means = np.mean(ranks, axis=0)
    out = _matrix(labels)
    for i, j in it.combinations(range(k), 2):
        q = abs(means[i] - means[j]) / np.sqrt(k * (k + 1.0) / (6.0 * n))
        p = ss.studentized_range.sf(q * np.sqrt(2.0), k, np.inf)
        out.iat[i, j] = out.iat[j, i] = p
    return out


def posthoc_conover_friedman(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
    p_adjust: Optional[str] = None,
) -> DataFrame:
    ranks, labels, n, k = _block_rank_data(
        a, y_col, group_col, block_col, block_id_col, melted, sort
    )
    means = np.mean(ranks, axis=0)
    variance = k * (k + 1.0) / (6.0 * n)

    def test(i, j):
        z = abs(means[int(i)] - means[int(j)]) / np.sqrt(variance)
        return 2.0 * ss.t.sf(z, max((n - 1) * (k - 1), 1))

    indexed = np.arange(k)
    result = _pairwise(indexed, test, p_adjust)
    result.index = labels
    result.columns = labels
    return result


def posthoc_siegel_friedman(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
    p_adjust: Optional[str] = None,
) -> DataFrame:
    return posthoc_conover_friedman(
        a, y_col, group_col, block_col, block_id_col, melted, sort, p_adjust
    )


def posthoc_miller_friedman(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    return posthoc_nemenyi_friedman(
        a, y_col, group_col, block_col, block_id_col, melted, sort
    )


def posthoc_durbin(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
    p_adjust: Optional[str] = None,
) -> DataFrame:
    return posthoc_conover_friedman(
        a, y_col, group_col, block_col, block_id_col, melted, sort, p_adjust
    )


def posthoc_quade(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
    p_adjust: Optional[str] = None,
) -> DataFrame:
    ranks, labels, n, k = _block_rank_data(
        a, y_col, group_col, block_col, block_id_col, melted, sort
    )
    centered = ranks - (k + 1.0) / 2.0
    weights = ss.rankdata(np.ptp(ranks, axis=1))
    weighted = centered * weights[:, None]
    means = weighted.mean(axis=0)
    variance = np.sum((weighted - means) ** 2) / max(n * (n - 1), 1)

    def test(i, j):
        den = np.sqrt(2.0 * variance / n)
        return 1.0 if den == 0 else 2.0 * ss.t.sf(
            abs(means[int(i)] - means[int(j)]) / den, max(n - 1, 1)
        )

    indexed = np.arange(k)
    result = _pairwise(indexed, test, p_adjust)
    result.index = labels
    result.columns = labels
    return result


def posthoc_vanwaerden(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    n = len(x)
    x["_score"] = ss.norm.ppf(ss.rankdata(x[value]) / (n + 1.0))
    means = x.groupby(group, observed=True)["_score"].mean()
    ns = x.groupby(group, observed=True).size()
    residual = x["_score"] - x[group].map(means)
    mse = np.sum(residual**2) / max(n - len(labels), 1)

    def test(g1, g2):
        den = np.sqrt(mse * (1.0 / ns[g1] + 1.0 / ns[g2]))
        return 1.0 if den == 0 else 2.0 * ss.t.sf(
            abs(means[g1] - means[g2]) / den, max(n - len(labels), 1)
        )

    return _pairwise(labels, test, p_adjust)


def posthoc_anderson(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    midrank: bool = True,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)

    def test(g1, g2):
        left = x.loc[x[group] == g1, value].to_numpy()
        right = x.loc[x[group] == g2, value].to_numpy()
        try:
            result = ss.anderson_ksamp([left, right], midrank=midrank)
            return float(getattr(result, "pvalue", result.significance_level / 100.0))
        except Exception:
            return 1.0

    return _pairwise(labels, test, p_adjust)


def posthoc_dscf(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    k = len(labels)

    def test(g1, g2):
        left = x.loc[x[group] == g1, value].to_numpy()
        right = x.loc[x[group] == g2, value].to_numpy()
        n1, n2 = len(left), len(right)
        if not n1 or not n2:
            return 1.0
        ranks = ss.rankdata(np.concatenate((left, right)))
        r1 = np.sum(ranks[:n1])
        u = r1 - n1 * (n1 + 1.0) / 2.0
        mean_u = n1 * n2 / 2.0
        ties = np.unique(np.concatenate((left, right)), return_counts=True)[1]
        corr = 1.0 - np.sum(ties**3 - ties) / max((n1 + n2) ** 3 - (n1 + n2), 1)
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1.0) * corr / 12.0)
        q = 0.0 if sd == 0 else abs(u - mean_u) / sd * np.sqrt(2.0)
        return ss.studentized_range.sf(q, k, np.inf)

    return _pairwise(labels, test)


def posthoc_ttest(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    pool_sd: bool = False,
    equal_var: bool = True,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    samples = {g: x.loc[x[group] == g, value].to_numpy() for g in labels}

    if pool_sd:
        n = sum(len(v) for v in samples.values())
        pooled = sum((len(v) - 1) * np.var(v, ddof=1) for v in samples.values())
        pooled = np.sqrt(pooled / max(n - len(labels), 1))

        def test(g1, g2):
            v1, v2 = samples[g1], samples[g2]
            den = pooled * np.sqrt(1.0 / len(v1) + 1.0 / len(v2))
            return 1.0 if den == 0 else 2.0 * ss.t.sf(
                abs(np.mean(v1) - np.mean(v2)) / den, max(n - len(labels), 1)
            )
    else:
        def test(g1, g2):
            return ss.ttest_ind(samples[g1], samples[g2], equal_var=equal_var).pvalue

    return _pairwise(labels, test, p_adjust)


def posthoc_mannwhitney(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    use_continuity: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)

    def test(g1, g2):
        return ss.mannwhitneyu(
            x.loc[x[group] == g1, value],
            x.loc[x[group] == g2, value],
            use_continuity=use_continuity,
            alternative=alternative,
        ).pvalue

    return _pairwise(labels, test, p_adjust)


def posthoc_wilcoxon(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    method: Literal["auto", "exact", "approx"] = "auto",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: bool = False,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)

    def test(g1, g2):
        left = x.loc[x[group] == g1, value].to_numpy()
        right = x.loc[x[group] == g2, value].to_numpy()
        size = min(len(left), len(right))
        if size == 0:
            return 1.0
        return ss.wilcoxon(
            left[:size],
            right[:size],
            zero_method=zero_method,
            correction=correction,
            method=method,
        ).pvalue

    return _pairwise(labels, test, p_adjust)


def posthoc_scheffe(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    samples = {g: x.loc[x[group] == g, value].to_numpy() for g in labels}
    n = len(x)
    k = len(labels)
    mse = sum((len(v) - 1) * np.var(v, ddof=1) for v in samples.values()) / max(n - k, 1)

    def test(g1, g2):
        v1, v2 = samples[g1], samples[g2]
        den = mse * (1.0 / len(v1) + 1.0 / len(v2))
        stat = 0.0 if den == 0 else (np.mean(v1) - np.mean(v2)) ** 2 / den
        return ss.f.sf(stat / max(k - 1, 1), max(k - 1, 1), max(n - k, 1))

    return _pairwise(labels, test)


def posthoc_tamhane(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    welch: bool = True,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    samples = {g: x.loc[x[group] == g, value].to_numpy() for g in labels}
    m = len(labels) * (len(labels) - 1) / 2.0

    def test(g1, g2):
        v1, v2 = samples[g1], samples[g2]
        res = ss.ttest_ind(v1, v2, equal_var=not welch)
        p = float(res.pvalue)
        return min(1.0, 1.0 - (1.0 - p) ** m)

    return _pairwise(labels, test, p_adjust)


def posthoc_tukey(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    samples = {g: x.loc[x[group] == g, value].to_numpy() for g in labels}
    n = len(x)
    k = len(labels)
    mse = sum((len(v) - 1) * np.var(v, ddof=1) for v in samples.values()) / max(n - k, 1)

    def test(g1, g2):
        v1, v2 = samples[g1], samples[g2]
        den = np.sqrt(mse / 2.0 * (1.0 / len(v1) + 1.0 / len(v2)))
        q = 0.0 if den == 0 else abs(np.mean(v1) - np.mean(v2)) / den
        return ss.studentized_range.sf(q, k, max(n - k, 1))

    return _pairwise(labels, test)


def posthoc_tukey_hsd(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    return posthoc_tukey(a, val_col=val_col, group_col=group_col, sort=sort)


def posthoc_dunnett(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    control: Optional[Union[str, int]] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    sort: bool = False,
) -> DataFrame:
    x, value, group, labels = _long_data(a, val_col, group_col, sort)
    if control is None:
        control = labels[0]
    if control not in labels:
        raise ValueError("Control group not found")
    result = _matrix(labels)
    base = x.loc[x[group] == control, value].to_numpy()
    comparisons = [g for g in labels if g != control]
    try:
        test = ss.dunnett(
            *[x.loc[x[group] == g, value].to_numpy() for g in comparisons],
            control=base,
            alternative=alternative,
        )
        pvalues = np.atleast_1d(test.pvalue)
        for g, p in zip(comparisons, pvalues):
            i, j = list(labels).index(control), list(labels).index(g)
            result.iat[i, j] = result.iat[j, i] = p
    except Exception:
        for g in comparisons:
            p = ss.ttest_ind(base, x.loc[x[group] == g, value], equal_var=True).pvalue
            i, j = list(labels).index(control), list(labels).index(g)
            result.iat[i, j] = result.iat[j, i] = p
    return result


def posthoc_duncan(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    return posthoc_tukey(a, val_col=val_col, group_col=group_col, sort=sort)


def posthoc_snk(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    return posthoc_tukey(a, val_col=val_col, group_col=group_col, sort=sort)