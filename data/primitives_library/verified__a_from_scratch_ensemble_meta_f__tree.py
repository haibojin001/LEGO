import math
from typing import Any, Iterable, List, Tuple

import numpy as np

__all__ = ["gini", "DecisionTree"]

_NAN_SENTINEL = object()
_GAIN_TOL = 1e-12


def _as_label_list(y) -> List[Any]:
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


def _as_1d_labels(y) -> np.ndarray:
    if y is None:
        return np.empty(0, dtype=object)
    if isinstance(y, (str, bytes)):
        arr = np.empty(1, dtype=object)
        arr[0] = y
        return arr
    arr = np.asarray(y, dtype=object)
    if arr.ndim == 0:
        out = np.empty(1, dtype=object)
        out[0] = arr.item()
        return out
    return arr.reshape(-1)


def _is_nan(value: Any) -> bool:
    try:
        result = math.isnan(value)
    except (TypeError, ValueError):
        try:
            result = np.isnan(value)
        except Exception:
            return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    if isinstance(result, np.ndarray):
        return bool(result.shape == () and bool(result))
    return False


def _labels_equal(left: Any, right: Any) -> bool:
    if _is_nan(left) and _is_nan(right):
        return True

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            try:
                return bool(np.array_equal(left, right))
            except Exception:
                return False
        except Exception:
            return False

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


def gini(y) -> float:
    labels = _as_label_list(y)
    counts, total = _label_counts(labels)
    if total <= 0:
        return 0.0

    impurity = 1.0
    denom = float(total)
    for count in counts:
        p = count / denom
        impurity -= p * p

    if impurity < 0.0 and impurity > -1e-15:
        impurity = 0.0
    return float(impurity)


def _gini_from_counts(counts, total=None) -> float:
    arr = np.asarray(counts, dtype=float)
    if total is None:
        total = float(arr.sum())
    else:
        total = float(total)

    if total <= 0.0:
        return 0.0

    probs = arr / total
    impurity = 1.0 - float(np.dot(probs, probs))
    if impurity < 0.0 and impurity > -1e-15:
        impurity = 0.0
    return float(impurity)


def _object_array(values) -> np.ndarray:
    arr = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


def _fallback_sort_key(value):
    if _is_nan(value):
        return (2, "", "")
    return (0, type(value).__name__, repr(value))


def _linear_contains(values, value) -> bool:
    for existing in values:
        if _labels_equal(existing, value):
            return True
    return False


def _unique_labels_unsorted(y) -> List[Any]:
    values = list(_as_1d_labels(y))
    unique = []
    seen = {}
    nan_seen = False

    for value in values:
        if _is_nan(value):
            if not nan_seen:
                unique.append(value)
                nan_seen = True
            continue

        try:
            if value in seen:
                continue
            seen[value] = True
            unique.append(value)
        except Exception:
            if not _linear_contains(unique, value):
                unique.append(value)

    return unique


def _unique_sorted_labels(y) -> List[Any]:
    unique = _unique_labels_unsorted(y)
    if not unique:
        return []

    non_nan = [value for value in unique if not _is_nan(value)]
    nan_values = [value for value in unique if _is_nan(value)]

    try:
        sorted_non_nan = sorted(non_nan)
    except Exception:
        sorted_non_nan = sorted(non_nan, key=_fallback_sort_key)

    return sorted_non_nan + nan_values[:1]


def _class_lookup(classes):
    lookup = {}
    fallback = []
    nan_index = None

    for i, cls in enumerate(classes):
        if _is_nan(cls):
            nan_index = i
            continue
        try:
            lookup[cls] = i
        except Exception:
            fallback.append((cls, i))

    return lookup, fallback, nan_index


def _encode_labels(y, classes) -> np.ndarray:
    labels = list(_as_1d_labels(y))
    encoded = np.empty(len(labels), dtype=int)
    lookup, fallback, nan_index = _class_lookup(classes)

    for i, label in enumerate(labels):
        found = -1

        if _is_nan(label):
            if nan_index is not None:
                found = nan_index
        else:
            try:
                found = lookup.get(label, -1)
            except Exception:
                found = -1

            if found == -1:
                for cls, code in fallback:
                    if _labels_equal(label, cls):
                        found = code
                        break

            if found == -1:
                for code, cls in enumerate(classes):
                    if _labels_equal(label, cls):
                        found = code
                        break

        if found == -1:
            raise ValueError("encountered a label not present in classes")
        encoded[i] = found

    return encoded


def _validate_X_fit(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 0:
        raise ValueError("X must be a 1D or 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")
    return arr


def _validate_X_predict(X, n_features) -> np.ndarray:
    arr = np.asarray(X, dtype=float)

    if arr.ndim == 0:
        if n_features == 1:
            return arr.reshape(1, 1)
        raise ValueError("X has the wrong number of features")

    if arr.ndim == 1:
        if n_features == 1:
            return arr.reshape(-1, 1)
        if arr.size == n_features:
            return arr.reshape(1, -1)
        raise ValueError("X has the wrong number of features")

    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")

    if arr.shape[1] != n_features:
        raise ValueError("X has the wrong number of features")

    return arr


def _threshold_between(a, b) -> float:
    a = float(a)
    b = float(b)

    if math.isnan(a) or math.isnan(b):
        return a

    if a == b:
        return a

    if math.isinf(a) or math.isinf(b):
        return a

    mid = a + (b - a) / 2.0
    if a < mid < b:
        return float(mid)

    mid = (a / 2.0) + (b / 2.0)
    if a < mid < b:
        return float(mid)

    return a


class _Node:
    __slots__ = (
        "prediction",
        "feature",
        "threshold",
        "nan_go_left",
        "left",
        "right",
        "n_samples",
        "counts",
    )

    def __init__(
        self,
        prediction=None,
        feature=None,
        threshold=None,
        nan_go_left=True,
        left=None,
        right=None,
        n_samples=0,
        counts=None,
    ):
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.nan_go_left = bool(nan_go_left)
        self.left = left
        self.right = right
        self.n_samples = int(n_samples)
        self.counts = counts

    @property
    def is_leaf(self):
        return self.feature is None


def _best_split(X, y_codes, n_classes, min_samples_leaf):
    n_samples, n_features = X.shape

    if n_samples < 2 * min_samples_leaf:
        return None

    parent_counts = np.bincount(y_codes, minlength=n_classes)
    parent_impurity = _gini_from_counts(parent_counts, n_samples)
    if parent_impurity <= 0.0:
        return None

    best_gain = 0.0
    best = None

    for feature in range(n_features):
        values = X[:, feature]
        nan_mask = np.isnan(values)
        valid_mask = ~nan_mask

        valid_n = int(np.sum(valid_mask))
        nan_n = n_samples - valid_n

        if valid_n == 0:
            continue

        valid_values = values[valid_mask]
        valid_y = y_codes[valid_mask]

        if nan_n:
            nan_counts = np.bincount(y_codes[nan_mask], minlength=n_classes)
        else:
            nan_counts = np.zeros(n_classes, dtype=int)

        if nan_n >= min_samples_leaf and valid_n >= min_samples_leaf:
            valid_counts = np.bincount(valid_y, minlength=n_classes)
            weighted = (
                (valid_n / n_samples) * _gini_from_counts(valid_counts, valid_n)
                + (nan_n / n_samples) * _gini_from_counts(nan_counts, nan_n)
            )
            gain = parent_impurity - weighted
            if gain > _GAIN_TOL and gain > best_gain + _GAIN_TOL:
                best_gain = gain
                best = (feature, float("inf"), False)

        if valid_n < 2:
            continue

        order = np.argsort(valid_values, kind="mergesort")
        sorted_values = valid_values[order]
        sorted_y = valid_y[order]

        left_non_counts = np.zeros(n_classes, dtype=int)
        right_non_counts = np.bincount(sorted_y, minlength=n_classes)

        pos = 0
        while pos < valid_n:
            current_value = sorted_values[pos]
            end = pos + 1
            while end < valid_n and sorted_values[end] == current_value:
                end += 1

            moved_counts = np.bincount(sorted_y[pos:end], minlength=n_classes)
            left_non_counts += moved_counts
            right_non_counts -= moved_counts

            if end >= valid_n:
                break

            next_value = sorted_values[end]
            threshold = _threshold_between(current_value, next_value)
            left_non_n = end
            right_non_n = valid_n - end

            left_n = left_non_n + nan_n
            right_n = right_non_n
            if left_n >= min_samples_leaf and right_n >= min_samples_leaf:
                left_counts = left_non_counts + nan_counts
                right_counts = right_non_counts
                weighted = (
                    (left_n / n_samples) * _gini_from_counts(left_counts, left_n)
                    + (right_n / n_samples) * _gini_from_counts(right_counts, right_n)
                )
                gain = parent_impurity - weighted
                if gain > _GAIN_TOL and gain > best_gain + _GAIN_TOL:
                    best_gain = gain
                    best = (feature, threshold, True)

            left_n = left_non_n
            right_n = right_non_n + nan_n
            if left_n >= min_samples_leaf and right_n >= min_samples_leaf:
                left_counts = left_non_counts
                right_counts = right_non_counts + nan_counts
                weighted = (
                    (left_n / n_samples) * _gini_from_counts(left_counts, left_n)
                    + (right_n / n_samples) * _gini_from_counts(right_counts, right_n)
                )
                gain = parent_impurity - weighted
                if gain > _GAIN_TOL and gain > best_gain + _GAIN_TOL:
                    best_gain = gain
                    best = (feature, threshold, False)

            pos = end

    return best


def _partition_mask(values, threshold, nan_go_left):
    nan_mask = np.isnan(values)
    mask = values <= threshold
    if np.any(nan_mask):
        mask = mask.copy()
        mask[nan_mask] = bool(nan_go_left)
    return mask


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        if max_depth is not None:
            depth = int(max_depth)
            if depth < 0 or depth != max_depth:
                raise ValueError("max_depth must be None or a non-negative integer")
            max_depth = depth

        leaf = int(min_samples_leaf)
        if leaf < 1 or leaf != min_samples_leaf:
            raise ValueError("min_samples_leaf must be a positive integer")

        self.max_depth = max_depth
        self.min_samples_leaf = leaf

    def fit(self, X, y):
        X_arr = _validate_X_fit(X)
        y_arr = _as_1d_labels(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if X_arr.shape[0] == 0:
            raise ValueError("cannot fit DecisionTree on an empty dataset")

        classes = _unique_sorted_labels(y_arr)
        if not classes:
            raise ValueError("y must contain at least one class")

        self.classes_ = _object_array(classes)
        self.n_classes_ = len(classes)
        self.n_features_in_ = X_arr.shape[1]

        y_codes = _encode_labels(y_arr, self.classes_)
        self._root = self._build(X_arr, y_codes, depth=0)
        return self

    def _build(self, X, y_codes, depth):
        n_samples = y_codes.shape[0]
        counts = np.bincount(y_codes, minlength=self.n_classes_)
        prediction_index = int(np.argmax(counts))

        node = _Node(
            prediction=self.classes_[prediction_index],
            n_samples=n_samples,
            counts=counts.copy(),
        )

        if np.count_nonzero(counts) <= 1:
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            return node

        if n_samples < 2 * self.min_samples_leaf:
            return node

        split = _best_split(X, y_codes, self.n_classes_, self.min_samples_leaf)
        if split is None:
            return node

        feature, threshold, nan_go_left = split
        left_mask = _partition_mask(X[:, feature], threshold, nan_go_left)
        left_n = int(np.sum(left_mask))
        right_n = n_samples - left_n

        if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
            return node

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.nan_go_left = bool(nan_go_left)
        node.left = self._build(X[left_mask], y_codes[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y_codes[~left_mask], depth + 1)
        return node

    def _check_is_fitted(self):
        if not hasattr(self, "_root") or not hasattr(self, "classes_"):
            raise ValueError("DecisionTree instance is not fitted yet")

    def _leaf_for_row(self, row):
        node = self._root
        while not node.is_leaf:
            value = row[node.feature]
            if np.isnan(value):
                go_left = node.nan_go_left
            else:
                go_left = value <= node.threshold
            node = node.left if go_left else node.right
        return node

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _validate_X_predict(X, self.n_features_in_)

        predictions = np.empty(X_arr.shape[0], dtype=object)
        for i in range(X_arr.shape[0]):
            predictions[i] = self._leaf_for_row(X_arr[i]).prediction

        return predictions

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _validate_X_predict(X, self.n_features_in_)

        proba = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)
        for i in range(X_arr.shape[0]):
            leaf = self._leaf_for_row(X_arr[i])
            counts = np.asarray(leaf.counts, dtype=float)
            total = float(counts.sum())
            if total > 0.0:
                proba[i, :] = counts / total

        return proba