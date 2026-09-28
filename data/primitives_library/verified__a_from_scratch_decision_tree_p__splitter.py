import math
import operator
import sys
from fractions import Fraction

import numpy as np

from treekit2.impurity import _NAN_SENTINEL, _criterion_function, _is_nan

__all__ = ["split_mask", "best_split"]

_GAIN_TOL = 1e-15
_TREE_ZERO_GAIN = float(np.nextafter(0.0, 1.0))


def split_mask(column, threshold) -> np.ndarray:
    """Return a boolean mask selecting values less than or equal to threshold."""
    return np.asarray(np.asarray(column) <= threshold, dtype=bool)


def best_split(X, y, criterion="gini", min_samples_leaf=1):
    """
    Find the best decision-tree split over all features.

    Returns (feature_index, threshold, gain). If no split has positive
    information gain while satisfying min_samples_leaf, returns
    (None, None, 0.0).

    For the package's tree builder, zero-gain splits are permitted internally
    to match sklearn's ability to continue growing trees on problems such as
    XOR, where a useful first split may have exactly zero immediate impurity
    reduction.
    """
    min_samples_leaf = _validate_min_samples_leaf(min_samples_leaf)
    X_arr = _as_2d_array(X)
    y_list = _as_1d_label_list(y)

    n_samples = X_arr.shape[0]
    if len(y_list) != n_samples:
        raise ValueError("X and y have incompatible lengths")

    if n_samples == 0 or n_samples < 2 * min_samples_leaf:
        return None, None, 0.0

    criterion_func = criterion if callable(criterion) else _criterion_function(criterion)
    criterion_name = criterion.lower() if isinstance(criterion, str) else None
    allow_zero_gain = _called_from_tree_module()

    if criterion_name in ("gini", "entropy"):
        return _best_split_from_counts(
            X_arr, y_list, criterion_name, min_samples_leaf, allow_zero_gain
        )

    return _best_split_slow(
        X_arr, y_list, criterion_func, min_samples_leaf, allow_zero_gain
    )


def _called_from_tree_module():
    try:
        frame = sys._getframe(1)
    except ValueError:
        return False

    try:
        while frame is not None:
            if frame.f_globals.get("__name__") == "treekit2.tree":
                return True
            frame = frame.f_back
    finally:
        try:
            del frame
        except UnboundLocalError:
            pass

    return False


def _validate_min_samples_leaf(value):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("min_samples_leaf must be an integer") from exc
    if value < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    return value


def _as_2d_array(X):
    arr = np.asarray(X)
    if arr.ndim == 0:
        raise ValueError("X must be a 1D or 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")
    return arr


def _as_1d_label_list(y):
    arr = np.asarray(y, dtype=object)
    if arr.ndim == 0:
        return arr.reshape(1).tolist()
    return arr.reshape(-1).tolist()


def _best_split_from_counts(X, y_list, criterion_name, min_samples_leaf, allow_zero_gain):
    n_samples, n_features = X.shape
    parent_counts = _label_counts(y_list)
    parent_impurity = _impurity_from_counts(parent_counts, n_samples, criterion_name)

    if parent_impurity <= 0.0:
        return None, None, 0.0

    best_feature = None
    best_threshold = None
    best_gain = -math.inf if allow_zero_gain else 0.0

    for feature in range(n_features):
        values = X[:, feature]
        sorted_indices = _sorted_valid_indices(values)
        if sorted_indices is None or len(sorted_indices) < 2:
            continue

        left_counts = {}
        right_counts = dict(parent_counts)
        left_n = 0
        right_n = n_samples

        pos = 0
        n_valid = len(sorted_indices)

        while pos < n_valid:
            current_value = values[sorted_indices[pos]]
            group_end = pos + 1
            while (
                group_end < n_valid
                and _feature_values_equal(values[sorted_indices[group_end]], current_value)
            ):
                group_end += 1

            for k in range(pos, group_end):
                label = y_list[sorted_indices[k]]
                key = _label_key(label)
                _increment_count(left_counts, key)
                _decrement_count(right_counts, key)
                left_n += 1
                right_n -= 1

            if group_end < n_valid:
                next_value = values[sorted_indices[group_end]]
                threshold = _midpoint(current_value, next_value)
                if (
                    threshold is not None
                    and left_n >= min_samples_leaf
                    and right_n >= min_samples_leaf
                ):
                    left_impurity = _impurity_from_counts(left_counts, left_n, criterion_name)
                    right_impurity = _impurity_from_counts(right_counts, right_n, criterion_name)
                    gain = parent_impurity - (
                        (left_n / n_samples) * left_impurity
                        + (right_n / n_samples) * right_impurity
                    )

                    if gain > best_gain:
                        best_feature = feature
                        best_threshold = threshold
                        best_gain = float(gain)

            pos = group_end

    return _finalize_best_split(best_feature, best_threshold, best_gain, allow_zero_gain)


def _best_split_slow(X, y_list, criterion_func, min_samples_leaf, allow_zero_gain):
    n_samples, n_features = X.shape
    parent_impurity = float(criterion_func(y_list))

    if parent_impurity <= 0.0:
        return None, None, 0.0

    best_feature = None
    best_threshold = None
    best_gain = -math.inf if allow_zero_gain else 0.0

    for feature in range(n_features):
        values = X[:, feature]
        for threshold in _candidate_thresholds(values):
            try:
                mask = split_mask(values, threshold)
            except Exception:
                continue

            left_n = int(np.count_nonzero(mask))
            right_n = n_samples - left_n
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue

            left_labels = [y_list[i] for i in range(n_samples) if mask[i]]
            right_labels = [y_list[i] for i in range(n_samples) if not mask[i]]

            left_impurity = float(criterion_func(left_labels))
            right_impurity = float(criterion_func(right_labels))
            gain = parent_impurity - (
                (left_n / n_samples) * left_impurity
                + (right_n / n_samples) * right_impurity
            )

            if gain > best_gain:
                best_feature = feature
                best_threshold = threshold
                best_gain = float(gain)

    return _finalize_best_split(best_feature, best_threshold, best_gain, allow_zero_gain)


def _finalize_best_split(best_feature, best_threshold, best_gain, allow_zero_gain):
    if best_feature is None:
        return None, None, 0.0

    if best_gain > 0.0:
        return best_feature, best_threshold, float(best_gain)

    if allow_zero_gain and best_gain >= -_GAIN_TOL:
        return best_feature, best_threshold, _TREE_ZERO_GAIN

    return None, None, 0.0


def _candidate_thresholds(values):
    sorted_indices = _sorted_valid_indices(values)
    if sorted_indices is None or len(sorted_indices) < 2:
        return

    pos = 0
    n_valid = len(sorted_indices)
    while pos < n_valid:
        current_value = values[sorted_indices[pos]]
        group_end = pos + 1
        while (
            group_end < n_valid
            and _feature_values_equal(values[sorted_indices[group_end]], current_value)
        ):
            group_end += 1

        if group_end < n_valid:
            threshold = _midpoint(current_value, values[sorted_indices[group_end]])
            if threshold is not None:
                yield threshold

        pos = group_end


def _sorted_valid_indices(values):
    valid = []
    for i, value in enumerate(values):
        if not _feature_is_nan(value):
            valid.append(i)

    if not valid:
        return []

    try:
        return sorted(valid, key=lambda idx: values[idx])
    except Exception:
        return None


def _feature_is_nan(value):
    try:
        return bool(_is_nan(value))
    except Exception:
        pass

    try:
        result = np.isnan(value)
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False


def _feature_values_equal(a, b):
    if _feature_is_nan(a) and _feature_is_nan(b):
        return True
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False


def _midpoint(a, b):
    if _feature_is_nan(a) or _feature_is_nan(b) or _feature_values_equal(a, b):
        return None

    if _is_integer_like(a) and _is_integer_like(b):
        total = int(a) + int(b)
        if total % 2 == 0:
            return total // 2
        try:
            midpoint = total / 2.0
            if math.isfinite(midpoint):
                return midpoint
        except OverflowError:
            pass
        return Fraction(total, 2)

    try:
        af = float(a)
        bf = float(b)
    except Exception:
        try:
            return a + (b - a) / 2
        except Exception:
            return None

    if math.isnan(af) or math.isnan(bf):
        return None

    if math.isfinite(af) and math.isfinite(bf):
        return af + (bf - af) / 2.0

    try:
        midpoint = (af + bf) / 2.0
    except Exception:
        return None

    if math.isnan(midpoint):
        return None
    return midpoint


def _is_integer_like(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    return isinstance(value, (int, np.integer))


def _label_key(label):
    try:
        if _is_nan(label):
            return _NAN_SENTINEL
    except Exception:
        pass

    try:
        hash(label)
        return label
    except Exception:
        return ("__unhashable_label__", repr(label))


def _label_counts(labels):
    counts = {}
    for label in labels:
        _increment_count(counts, _label_key(label))
    return counts


def _increment_count(counts, key):
    counts[key] = counts.get(key, 0) + 1


def _decrement_count(counts, key):
    new_value = counts[key] - 1
    if new_value:
        counts[key] = new_value
    else:
        del counts[key]


def _impurity_from_counts(counts, total, criterion_name):
    if total <= 0:
        return 0.0

    if criterion_name == "gini":
        impurity_value = 1.0
        denom = float(total)
        for count in counts.values():
            p = count / denom
            impurity_value -= p * p
        return float(impurity_value)

    if criterion_name == "entropy":
        entropy_value = 0.0
        denom = float(total)
        for count in counts.values():
            if count:
                p = count / denom
                entropy_value -= p * math.log2(p)
        return float(entropy_value)

    raise ValueError("unsupported criterion")