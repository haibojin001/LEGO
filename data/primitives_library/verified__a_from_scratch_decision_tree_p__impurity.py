"""Impurity criteria for classification labels."""

import math
from typing import Any, Iterable, List, Tuple

import numpy as np

__all__ = ["gini", "entropy", "impurity", "information_gain"]


_NAN_SENTINEL = object()


def _as_label_list(y) -> List[Any]:
    """Return labels as a reusable list without treating strings as iterables."""
    if y is None:
        return []

    if isinstance(y, np.ndarray):
        if y.ndim == 0:
            return [y.item()]
        return y.ravel().tolist()

    if isinstance(y, (str, bytes)):
        return [y]

    try:
        return list(y)
    except TypeError:
        return [y]


def _is_nan(value: Any) -> bool:
    """True for scalar NaN-like values; False for arrays and non-numeric labels."""
    try:
        result = math.isnan(value)
    except (TypeError, ValueError):
        return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _labels_equal(left: Any, right: Any) -> bool:
    """Equality helper for fallback counting of unhashable labels."""
    if _is_nan(left) and _is_nan(right):
        return True

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))

    try:
        result = left == right
    except Exception:
        return False

    if isinstance(result, np.ndarray):
        if result.shape == ():
            return bool(result.item())
        return bool(np.all(result))

    try:
        return bool(result)
    except Exception:
        return False


def _label_counts(labels: Iterable[Any]) -> Tuple[List[int], int]:
    """Count labels, including a safe fallback for unhashable labels."""
    hashed_counts = {}
    fallback_values = []
    fallback_counts = []
    total = 0

    for label in labels:
        total += 1

        try:
            key = _NAN_SENTINEL if _is_nan(label) else label
            hashed_counts[key] = hashed_counts.get(key, 0) + 1
            continue
        except TypeError:
            pass

        found = False
        for i, existing in enumerate(fallback_values):
            if _labels_equal(existing, label):
                fallback_counts[i] += 1
                found = True
                break

        if not found:
            fallback_values.append(label)
            fallback_counts.append(1)

    return list(hashed_counts.values()) + fallback_counts, total


def _criterion_function(criterion):
    if isinstance(criterion, str):
        name = criterion.strip().lower()
        if name == "gini":
            return gini
        if name == "entropy":
            return entropy

    raise ValueError(f"Unknown impurity criterion: {criterion!r}")


def gini(y) -> float:
    """Compute Gini impurity, 1 - sum(p_i ** 2), for a label distribution."""
    labels = _as_label_list(y)
    counts, total = _label_counts(labels)

    if total == 0:
        return 0.0

    score = 1.0
    total_float = float(total)
    for count in counts:
        probability = count / total_float
        score -= probability * probability

    return float(score)


def entropy(y) -> float:
    """Compute Shannon entropy, -sum(p_i * log2(p_i)), for a label distribution."""
    labels = _as_label_list(y)
    counts, total = _label_counts(labels)

    if total == 0:
        return 0.0

    score = 0.0
    total_float = float(total)
    for count in counts:
        if count:
            probability = count / total_float
            score -= probability * math.log2(probability)

    return float(score)


def impurity(y, criterion='gini') -> float:
    """Dispatch to an impurity criterion by name."""
    return float(_criterion_function(criterion)(y))


def information_gain(y, y_left, y_right, criterion='gini') -> float:
    """Compute impurity reduction from splitting y into y_left and y_right."""
    criterion_fn = _criterion_function(criterion)

    labels = _as_label_list(y)
    left_labels = _as_label_list(y_left)
    right_labels = _as_label_list(y_right)

    total = len(labels)
    if total == 0:
        return 0.0

    parent_impurity = criterion_fn(labels)
    left_impurity = criterion_fn(left_labels)
    right_impurity = criterion_fn(right_labels)

    weighted_children = (
        (len(left_labels) / total) * left_impurity
        + (len(right_labels) / total) * right_impurity
    )

    return float(parent_impurity - weighted_children)