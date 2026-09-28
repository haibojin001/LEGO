"""Impurity utilities for classification tree splits."""

import math
import operator
from typing import Any, Iterable, List, Tuple

import numpy as np

__all__ = ["gini", "best_split"]

_NAN_SENTINEL = object()
_GAIN_TOL = 1e-15


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


def _as_1d_label_list(y) -> List[Any]:
    """Return labels flattened to one dimension."""
    if y is None:
        return []

    if isinstance(y, (str, bytes)):
        return [y]

    arr = np.asarray(y, dtype=object)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.reshape(-1).tolist()


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


def _encode_labels(labels: List[Any]):
    """Encode arbitrary labels as integer class ids and return codes and counts."""
    hashed_codes = {}
    fallback_values = []
    fallback_codes = []
    codes = []
    counts = []

    for label in labels:
        try:
            key = _NAN_SENTINEL if _is_nan(label) else label
            code = hashed_codes.get(key, -1)
            if code == -1:
                code = len(counts)
                hashed_codes[key] = code
                counts.append(0)
            counts[code] += 1
            codes.append(code)
            continue
        except TypeError:
            pass

        found_code = -1
        for i, existing in enumerate(fallback_values):
            if _labels_equal(existing, label):
                found_code = fallback_codes[i]
                break

        if found_code == -1:
            found_code = len(counts)
            fallback_values.append(label)
            fallback_codes.append(found_code)
            counts.append(0)

        counts[found_code] += 1
        codes.append(found_code)

    return np.asarray(codes, dtype=np.intp), np.asarray(counts, dtype=np.int64)


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

    if score < 0.0 and score > -1e-15:
        score = 0.0
    return float(score)


def _validate_min_samples_leaf(value) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("min_samples_leaf must be an integer") from exc

    if value < 1:
        raise ValueError("min_samples_leaf must be at least 1")

    return int(value)


def _as_2d_numeric_array(X) -> np.ndarray:
    try:
        arr = np.asarray(X)
    except ValueError as exc:
        raise ValueError("X must be a 1D or 2D numeric array-like object") from exc

    if arr.ndim == 0:
        raise ValueError("X must be a 1D or 2D numeric array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D numeric array-like object")

    try:
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric values") from exc

    return arr


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _normalize_feature_indices(feature_indices, n_features: int) -> List[int]:
    if feature_indices is None:
        return list(range(n_features))

    if _is_bool_like(feature_indices):
        raise ValueError("feature_indices must be integer indices or a boolean mask")

    if isinstance(feature_indices, np.ndarray):
        if feature_indices.ndim == 0:
            seq = [feature_indices.item()]
        elif feature_indices.dtype == np.bool_:
            if feature_indices.ndim != 1 or feature_indices.shape[0] != n_features:
                raise ValueError("boolean feature mask must be one-dimensional with length n_features")
            return [int(i) for i in np.flatnonzero(feature_indices)]
        else:
            if feature_indices.ndim != 1:
                raise ValueError("feature_indices must be one-dimensional")
            seq = feature_indices.tolist()
    else:
        try:
            seq = list(feature_indices)
        except TypeError:
            seq = [feature_indices]

    if len(seq) == 0:
        return []

    if all(_is_bool_like(item) for item in seq):
        if len(seq) != n_features:
            raise ValueError("boolean feature mask must have length n_features")
        return [i for i, flag in enumerate(seq) if bool(flag)]

    features = []
    seen = set()
    for raw_index in seq:
        if _is_bool_like(raw_index):
            raise ValueError("boolean values are not valid feature indices unless used as a full boolean mask")

        try:
            index = operator.index(raw_index)
        except TypeError as exc:
            raise ValueError("feature_indices must contain integers") from exc

        index = int(index)
        if index < 0:
            index += n_features

        if index < 0 or index >= n_features:
            raise ValueError("feature index out of range")

        if index not in seen:
            seen.add(index)
            features.append(index)

    return features


def _gini_from_sumsq(sum_squares: float, n: int) -> float:
    if n <= 0:
        return 0.0
    impurity = 1.0 - (float(sum_squares) / (float(n) * float(n)))
    if impurity < 0.0 and impurity > -1e-15:
        impurity = 0.0
    return float(impurity)


def _safe_threshold(left_value: float, right_value: float) -> float:
    threshold = left_value + (right_value - left_value) / 2.0

    if not np.isfinite(threshold) or not (left_value < threshold < right_value):
        threshold = left_value

    return float(threshold)


def best_split(X, y, feature_indices=None, min_samples_leaf=1):
    """
    Find the best Gini-impurity split.

    Searches numeric threshold splits of the form X[:, feature] <= threshold.
    If feature_indices is provided, only those columns are considered.

    Returns:
        (feature, threshold, gain), or (None, None, 0.0) if no positive-gain
        split satisfies min_samples_leaf.
    """
    min_samples_leaf = _validate_min_samples_leaf(min_samples_leaf)
    X_arr = _as_2d_numeric_array(X)
    y_list = _as_1d_label_list(y)

    n_samples, n_features = X_arr.shape
    if len(y_list) != n_samples:
        raise ValueError("X and y have incompatible lengths")

    if n_samples == 0 or n_samples < 2 * min_samples_leaf:
        return None, None, 0.0

    features = _normalize_feature_indices(feature_indices, n_features)
    if not features:
        return None, None, 0.0

    label_codes, parent_counts = _encode_labels(y_list)
    if parent_counts.size <= 1:
        return None, None, 0.0

    parent_sum_squares = float(sum(float(count) * float(count) for count in parent_counts))
    parent_impurity = _gini_from_sumsq(parent_sum_squares, n_samples)

    if parent_impurity <= 0.0:
        return None, None, 0.0

    best_feature = None
    best_threshold = None
    best_gain = 0.0
    n_classes = parent_counts.size
    n_samples_float = float(n_samples)

    for feature in features:
        column = X_arr[:, feature]
        finite_mask = np.isfinite(column)
        n_finite = int(np.sum(finite_mask))

        if n_finite < min_samples_leaf:
            continue

        finite_indices = np.flatnonzero(finite_mask)
        if n_finite < 2:
            continue

        order_within_finite = np.argsort(column[finite_indices], kind="mergesort")
        order = finite_indices[order_within_finite]
        sorted_values = column[order]
        sorted_codes = label_codes[order]

        left_counts = np.zeros(n_classes, dtype=np.int64)
        right_counts = parent_counts.copy()
        left_n = 0
        right_n = n_samples
        left_sum_squares = 0.0
        right_sum_squares = parent_sum_squares

        pos = 0
        while pos < n_finite:
            current_value = sorted_values[pos]

            while pos < n_finite and sorted_values[pos] == current_value:
                code = int(sorted_codes[pos])

                old_left = int(left_counts[code])
                left_sum_squares += 2.0 * old_left + 1.0
                left_counts[code] = old_left + 1

                old_right = int(right_counts[code])
                right_sum_squares += -2.0 * old_right + 1.0
                right_counts[code] = old_right - 1

                left_n += 1
                right_n -= 1
                pos += 1

            if pos >= n_finite:
                break

            next_value = sorted_values[pos]
            if not current_value < next_value:
                continue

            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue

            left_impurity = _gini_from_sumsq(left_sum_squares, left_n)
            right_impurity = _gini_from_sumsq(right_sum_squares, right_n)

            weighted_impurity = (
                (float(left_n) / n_samples_float) * left_impurity
                + (float(right_n) / n_samples_float) * right_impurity
            )
            gain = parent_impurity - weighted_impurity

            if gain > best_gain + _GAIN_TOL:
                best_feature = int(feature)
                best_threshold = _safe_threshold(float(current_value), float(next_value))
                best_gain = float(gain)

    if best_feature is None:
        return None, None, 0.0

    return best_feature, best_threshold, float(best_gain)