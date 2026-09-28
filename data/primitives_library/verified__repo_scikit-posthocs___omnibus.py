import warnings
import itertools as it
from typing import List, Optional, Union, cast

import numpy as np
import scipy.stats as ss
from numpy.typing import ArrayLike
from pandas import Categorical, DataFrame, Series

from scikit_posthocs._posthocs import (
    __complete_block_matrix,
    __convert_to_block_df,
    __convert_to_df,
)


def test_mackwolfe(
    data: Union[ArrayLike, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p: Optional[int] = None,
    n_perm: int = 100,
    sort: bool = False,
) -> tuple[float, float]:
    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(
            x[_group_col],
            categories=x[_group_col].unique(),
            ordered=True,
        )

    x.sort_values(by=[_group_col], ascending=True, inplace=True)

    k = x[_group_col].unique().size

    if p is not None and not 0 <= p < k:
        raise ValueError(f"p must be between 0 and {k - 1}; got {p}")

    ranks = x[_val_col].rank()
    sizes = cast(Series, x.groupby(_group_col, observed=True)[_val_col].count())
    levels = x[_group_col].unique()
    positions = [
        np.flatnonzero(x[_group_col].to_numpy() == level)
        for level in levels
    ]

    def count_greater(left: np.ndarray, right: np.ndarray) -> float:
        ordered = np.sort(right)
        return float(
            np.sum(ordered.size - np.searchsorted(ordered, left, side="right"))
        )

    def u_matrix(rank_values: np.ndarray) -> np.ndarray:
        grouped = [rank_values[index] for index in positions]
        result = np.identity(k)

        for i in range(k):
            for j in range(i):
                result[i, j] = count_greater(grouped[i], grouped[j])
                result[j, i] = count_greater(grouped[j], grouped[i])

        return result

    def umbrella_value(peak: int, matrix: np.ndarray) -> float:
        increasing = np.triu(matrix[: peak + 1, : peak + 1], k=1).sum()
        decreasing = np.tril(matrix[peak:, peak:], k=-1).sum()
        return float(increasing + decreasing)

    def first_total(peak: int, counts: Series) -> float:
        return float(np.sum(counts[: peak + 1]))

    def second_total(peak: int, counts: Series) -> float:
        return float(np.sum(counts[peak:k]))

    def expected_value(peak: int, counts: Series) -> float:
        n1 = first_total(peak, counts)
        n2 = second_total(peak, counts)
        return float(
            (
                n1**2.0
                + n2**2.0
                - np.sum(counts**2.0)
                - counts.iloc[peak] ** 2.0
            )
            / 4.0
        )

    def variance(peak: int, counts: Series) -> float:
        n1 = first_total(peak, counts)
        n2 = second_total(peak, counts)
        total = np.sum(counts)
        peak_count = counts.iloc[peak]

        return float(
            (
                2.0 * (n1**3 + n2**3)
                + 3.0 * (n1**2 + n2**2)
                - np.sum(counts**2.0 * (2.0 * counts + 3.0))
                - peak_count**2.0 * (2.0 * peak_count + 3.0)
                + 12.0 * peak_count * n1 * n2
                - 12.0 * peak_count**2.0 * total
            )
            / 72.0
        )

    if p is not None:
        matrix = u_matrix(ranks.to_numpy())
        estimate = umbrella_value(p, matrix)
        mean = expected_value(p, sizes)
        sd = np.sqrt(variance(p, sizes))
        statistic = (estimate - mean) / sd
        p_value = ss.norm.sf(statistic).item()
    else:
        rank_values = ranks.to_numpy()
        matrix = u_matrix(rank_values)
        observed = np.array(
            [umbrella_value(i, matrix) for i in range(k)]
        ).ravel()
        means = np.array(
            [expected_value(i, sizes) for i in range(k)]
        ).ravel()
        variances = np.array(
            [variance(i, sizes) for i in range(k)]
        ).ravel()

        standardized = (observed - means) / np.sqrt(variances)
        statistic = float(np.max(standardized))

        simulated = []
        for _ in range(n_perm):
            permutation = np.random.permutation(rank_values)
            matrix = u_matrix(permutation)
            values = np.array(
                [umbrella_value(i, matrix) for i in range(k)]
            )
            standardized_values = (values - means) / np.sqrt(variances)
            simulated.append(np.max(standardized_values))

        simulated_array = np.asarray(simulated)
        p_value = simulated_array[simulated_array > statistic].size / n_perm

    return p_value, statistic


def test_osrt(
    data: Union[List, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> tuple[float, float, int]:
    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(
            x[_group_col],
            categories=x[_group_col].unique(),
            ordered=True,
        )

    x.sort_values(by=[_group_col], ascending=True, inplace=True)

    groups = np.unique(x[_group_col])
    grouped = x.groupby(_group_col, observed=True)[_val_col]

    means = grouped.mean()
    counts = grouped.count()
    k = groups.size
    n = len(x.index)
    df = n - k

    residuals = x[_val_col] - grouped.transform("mean")
    sigma = np.sqrt(np.sum(residuals**2.0) / df)

    def comparison(i: int, j: int) -> float:
        difference = means.loc[groups[j]] - means.loc[groups[i]]
        denominator = (
            sigma
            / np.sqrt(2.0)
            * np.sqrt(1.0 / counts[groups[j]] + 1.0 / counts[groups[i]])
        )
        return float(np.abs(difference) / denominator)

    values = np.zeros((k, k), dtype=float)

    for i, j in it.combinations(range(k), 2):
        values[i, j] = comparison(i, j)

    statistic = float(np.max(values))
    p_value = ss.studentized_range.sf(statistic, k, df)

    return p_value, statistic, df


def test_durbin(
    data: Union[List, np.ndarray, DataFrame],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = True,
) -> tuple[float, float, int]:
    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        data,
        y_col,
        group_col,
        block_col,
        block_id_col,
        melted,
    )

    if not sort:
        x[_group_col] = Categorical(
            x[_group_col],
            categories=x[_group_col].unique(),
            ordered=True,
        )
        x[_block_col] = Categorical(
            x[_block_col],
            categories=x[_block_col].unique(),
            ordered=True,
        )

    x.sort_values(by=[_block_col, _group_col], ascending=True, inplace=True)

    treatments = x[_group_col].unique()
    blocks = x[_block_id_col].unique()

    t = treatments.size
    b = blocks.size

    block_sizes = x.groupby(_block_id_col, observed=True)[_y_col].count()
    treatment_sizes = x.groupby(_group_col, observed=True)[_y_col].count()

    r = block_sizes.iloc[0]
    k = treatment_sizes.iloc[0]

    if not (block_sizes == r).all() or not (treatment_sizes == k).all():
        warnings.warn(
            "Durbin's test requires a balanced incomplete block design.",
            RuntimeWarning,
            stacklevel=2,
        )

    x["_rank"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    rank_sums = x.groupby(_group_col, observed=True)["_rank"].sum()

    a_value = np.sum(x["_rank"] ** 2.0)
    c_value = b * r * (r + 1.0) ** 2.0 / 4.0

    statistic = float(
        (t - 1.0)
        / (a_value - c_value)
        * (np.sum(rank_sums**2.0) / k - c_value)
    )
    df = t - 1
    p_value = ss.chi2.sf(statistic, df)

    return p_value, statistic, df


def test_quade(
    data: Union[List, np.ndarray, DataFrame],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = False,
) -> tuple[float, float, int]:
    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        data,
        y_col,
        group_col,
        block_col,
        block_id_col,
        melted,
    )

    x = __complete_block_matrix(
        x,
        _y_col,
        _group_col,
        _block_col,
        _block_id_col,
    )

    if not sort:
        x[_group_col] = Categorical(
            x[_group_col],
            categories=x[_group_col].unique(),
            ordered=True,
        )
        x[_block_col] = Categorical(
            x[_block_col],
            categories=x[_block_col].unique(),
            ordered=True,
        )

    x.sort_values(by=[_block_col, _group_col], ascending=True, inplace=True)

    groups = x[_group_col].unique()
    blocks = x[_block_id_col].unique()

    k = groups.size
    b = blocks.size

    x["_rank"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()

    ranges = x.groupby(_block_id_col, observed=True)[_y_col].transform(
        lambda values: values.max() - values.min()
    )
    x["_range_rank"] = ranges.rank()

    x["_q"] = (
        x["_rank"] - (k + 1.0) / 2.0
    ) * x["_range_rank"]

    group_sums = x.groupby(_group_col, observed=True)["_q"].sum()
    q_squared = np.sum(x["_q"] ** 2.0)
    treatment_ss = np.sum(group_sums**2.0) / b
    residual_ss = q_squared - treatment_ss

    df = k - 1
    denominator_df = (b - 1) * df
    statistic = float((b - 1.0) * treatment_ss / residual_ss)
    p_value = ss.f.sf(statistic, df, denominator_df)

    return p_value, statistic, df


def test_jonckheere(
    data: Union[ArrayLike, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    alternative: str = "increasing",
    sort: bool = False,
) -> tuple[float, float]:
    if alternative not in ("increasing", "decreasing"):
        raise ValueError(
            "alternative must be either 'increasing' or 'decreasing'; "
            f"got {alternative!r}"
        )

    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(
            x[_group_col],
            categories=x[_group_col].unique(),
            ordered=True,
        )

    x.sort_values(by=[_group_col], ascending=True, inplace=True)

    levels = x[_group_col].unique()
    k = levels.size
    grouped_values = [
        x.loc[x[_group_col] == level, _val_col].to_numpy()
        for level in levels
    ]
    counts = np.asarray([values.size for values in grouped_values], dtype=float)

    statistic = 0.0
    for i in range(k):
        for j in range(i):
            later = grouped_values[i]
            earlier = grouped_values[j]
            differences = later[:, None] - earlier[None, :]
            statistic += float(np.sum(differences > 0.0))
            statistic += 0.5 * float(np.sum(differences == 0.0))

    total = float(np.sum(counts))
    mean = (total**2.0 - np.sum(counts**2.0)) / 4.0
    variance = (
        total**2.0 * (2.0 * total + 3.0)
        - np.sum(counts**2.0 * (2.0 * counts + 3.0))
    ) / 72.0

    z_value = (statistic - mean) / np.sqrt(variance)

    if alternative == "increasing":
        p_value = ss.norm.sf(z_value).item()
    else:
        p_value = ss.norm.cdf(z_value).item()

    return p_value, float(z_value)