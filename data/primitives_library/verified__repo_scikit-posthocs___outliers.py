from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t


def outliers_iqr(
    x: Union[List, np.ndarray], ret: str = "filtered", coef: float = 1.5
) -> np.ndarray:
    arr = np.asarray(x)
    lower_quartile, upper_quartile = np.percentile(arr, (25, 75))
    spread = upper_quartile - lower_quartile
    lower_bound = lower_quartile - coef * spread
    upper_bound = upper_quartile + coef * spread
    inside = (arr >= lower_bound) & (arr <= upper_bound)

    if ret == "indices":
        return np.where(inside)[0]
    if ret == "outliers":
        return arr[~inside]
    if ret == "outliers_indices":
        return np.where(~inside)[0]
    return arr[inside]


def outliers_grubbs(
    x: Union[List, np.ndarray], hypo: bool = False, alpha: float = 0.05
) -> Union[np.ndarray, bool]:
    arr = np.copy(x)
    distances = np.abs(arr - np.mean(arr))
    candidate = np.argmax(distances)
    statistic = distances[candidate] / np.std(arr, ddof=1)
    sample_size = len(arr)
    critical_t = t.ppf(1.0 - alpha / (2.0 * sample_size), sample_size - 2)
    critical_value = (
        (sample_size - 1)
        / np.sqrt(sample_size)
        * np.sqrt(
            critical_t**2 / (sample_size - 2 + critical_t**2)
        )
    )
    rejected = statistic > critical_value

    if hypo:
        return rejected
    if rejected:
        return np.delete(arr, candidate)
    return arr


def outliers_tietjen(
    x: Union[List, np.ndarray], k: int, hypo: bool = False, alpha: float = 0.05
) -> Union[np.ndarray, bool]:
    arr = np.copy(x)
    sample_size = arr.size

    def statistic(values: np.ndarray, count: int) -> float:
        center = values.mean()
        ordered = values[np.abs(values - center).argsort()]
        retained = ordered[:-count]
        return np.sum((retained - retained.mean()) ** 2) / np.sum(
            (ordered - center) ** 2
        )

    observed = statistic(arr, k)
    iterations = 10000
    simulated = np.empty(iterations)

    if 0 < k < sample_size:
        retained_count = sample_size - k
        chunk = 256
        for first in range(0, iterations, chunk):
            last = min(first + chunk, iterations)
            random_values = np.random.normal(size=(last - first, sample_size))
            means = random_values.mean(axis=1, keepdims=True)
            distances = np.abs(random_values - means)
            selected = np.argpartition(
                distances, retained_count - 1, axis=1
            )[:, :retained_count]
            retained = np.take_along_axis(random_values, selected, axis=1)
            numerator = np.sum(
                (retained - retained.mean(axis=1, keepdims=True)) ** 2,
                axis=1,
            )
            denominator = np.sum((random_values - means) ** 2, axis=1)
            simulated[first:last] = numerator / denominator
    else:
        for index in range(iterations):
            simulated[index] = statistic(np.random.normal(size=sample_size), k)

    threshold = np.percentile(simulated, alpha * 100.0)
    rejected = observed < threshold

    if hypo:
        return rejected
    if rejected:
        candidates = np.argpartition(np.abs(arr - arr.mean()), -k)[-k:]
        return np.delete(arr, candidates)
    return arr


def outliers_gesd(
    x: ArrayLike,
    outliers: int = 5,
    hypo: bool = False,
    report: bool = False,
    alpha: float = 0.05,
) -> np.ndarray:
    original = np.copy(x)
    values = np.copy(x)
    sample_size = len(values)

    statistics = np.zeros(outliers)
    critical_values = np.zeros(outliers)
    original_indices = np.arange(np.asarray(values).size)

    removed_indices = np.zeros(outliers, dtype=int)

    for step in range(outliers):
        mean = np.mean(values)
        distances = np.abs(values - mean)
        local_index = np.argmax(distances)

        statistics[step] = distances[local_index] / np.std(values, ddof=1)

        remaining = sample_size - step
        probability = 1.0 - alpha / (2.0 * remaining)
        quantile = t.ppf(probability, remaining - 2)
        critical_values[step] = (
            (remaining - 1)
            * quantile
            / np.sqrt((remaining - 2 + quantile**2) * remaining)
        )

        removed_indices[step] = original_indices[local_index]
        values = np.delete(values, local_index)
        original_indices = np.delete(original_indices, local_index)

    decisions = statistics > critical_values

    if report:
        detected = (
            int(np.max(np.where(decisions)[0]) + 1) if np.any(decisions) else 0
        )
        print("H0: no outliers in the data")
        print("Ha: up to {} outliers in the data".format(outliers))
        print("")
        print("Significance level: alpha = {}".format(alpha))
        print("Critical region: R > lambda")
        print("")
        print("Number of outliers: {}".format(detected))
        print("{:<10}{:<15}{:<15}{:<10}".format("Sample", "R", "lambda", "Outlier"))
        for step, (value, critical, decision) in enumerate(
            zip(statistics, critical_values, decisions), start=1
        ):
            print(
                "{:<10}{:<15.6f}{:<15.6f}{:<10}".format(
                    step, value, critical, bool(decision)
                )
            )

    if hypo:
        return decisions

    if np.any(decisions):
        count = int(np.max(np.where(decisions)[0]) + 1)
        return np.delete(original, removed_indices[:count])

    return original