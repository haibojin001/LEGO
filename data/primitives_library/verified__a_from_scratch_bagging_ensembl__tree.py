import math
from typing import Any, List

import numpy as np


__all__ = ["gini", "best_split", "DecisionTree"]


_GAIN_TOL = 1e-12


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


def _is_nan(value) -> bool:
    try:
        result = np.isnan(value)
        if isinstance(result, np.ndarray):
            return bool(result.shape == () and bool(result))
        return bool(result)
    except Exception:
        return False


def _labels_equal(a, b) -> bool:
    if _is_nan(a) and _is_nan(b):
        return True
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False


def _object_array(values):
    arr = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


def _fallback_sort_key(value):
    if _is_nan(value):
        return (2, "", "")
    return (0, type(value).__name__, repr(value))


def _linear_contains(values, value):
    for existing in values:
        if _labels_equal(value, existing):
            return True
    return False


def _unique_labels_unsorted(y) -> List[Any]:
    values = list(np.asarray(y, dtype=object).reshape(-1))
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

    non_nan = [v for v in unique if not _is_nan(v)]
    nan_values = [v for v in unique if _is_nan(v)]

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
    labels = list(np.asarray(y, dtype=object).reshape(-1))
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

            if found < 0:
                for cls, idx in fallback:
                    if _labels_equal(label, cls):
                        found = idx
                        break

            if found < 0:
                for idx, cls in enumerate(classes):
                    if _labels_equal(label, cls):
                        found = idx
                        break

        if found < 0:
            raise ValueError("y contains a label not present in classes")
        encoded[i] = int(found)

    return encoded


def _as_training_array(X) -> np.ndarray:
    try:
        arr = np.asarray(X, dtype=float)
    except Exception as exc:
        raise ValueError("X must be a numeric array-like object") from exc

    if arr.ndim == 0:
        raise ValueError("X must be a 1D or 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")

    return np.asarray(arr, dtype=float)


def _as_prediction_array(X, n_features) -> np.ndarray:
    try:
        raw = np.asarray(X, dtype=float)
    except Exception as exc:
        raise ValueError("X must be a numeric array-like object") from exc

    if raw.ndim == 0:
        raise ValueError("X must be a 1D or 2D array-like object")

    if raw.ndim == 1:
        if n_features is not None and raw.size == int(n_features):
            return raw.reshape(1, int(n_features))
        return raw.reshape(-1, 1)

    if raw.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")

    return np.asarray(raw, dtype=float)


def _as_label_array(y) -> np.ndarray:
    arr = np.asarray(y, dtype=object)
    if arr.ndim == 0:
        raise ValueError("y must be a 1D array-like object")
    return arr.reshape(-1)


def _validate_min_samples_leaf(min_samples_leaf) -> int:
    try:
        value = int(min_samples_leaf)
    except Exception as exc:
        raise ValueError("min_samples_leaf must be a positive integer") from exc
    if value < 1:
        raise ValueError("min_samples_leaf must be a positive integer")
    return value


def _gini_from_counts(counts) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    return float(1.0 - np.dot(probs, probs))


def _label_counts(y):
    labels = list(np.asarray(y, dtype=object).reshape(-1))
    counts = []
    lookup = {}
    fallback = []
    nan_index = None

    for label in labels:
        if _is_nan(label):
            if nan_index is None:
                nan_index = len(counts)
                counts.append(1.0)
            else:
                counts[nan_index] += 1.0
            continue

        found = None
        try:
            found = lookup.get(label, None)
        except Exception:
            found = None

        if found is not None:
            counts[found] += 1.0
            continue

        for stored, idx in fallback:
            if _labels_equal(label, stored):
                found = idx
                break

        if found is not None:
            counts[found] += 1.0
            continue

        idx = len(counts)
        counts.append(1.0)
        try:
            lookup[label] = idx
        except Exception:
            fallback.append((label, idx))

    return np.asarray(counts, dtype=float)


def gini(y) -> float:
    labels = _as_label_array(y)
    if labels.size == 0:
        return 0.0
    return _gini_from_counts(_label_counts(labels))


def _threshold_between(a, b):
    a = float(a)
    b = float(b)

    if math.isfinite(a) and math.isfinite(b):
        threshold = a + (b - a) / 2.0
        if math.isinf(threshold):
            threshold = a / 2.0 + b / 2.0
    elif math.isneginf(a) and math.isposinf(b):
        threshold = 0.0
    elif math.isneginf(a):
        threshold = a
    elif math.isposinf(b):
        threshold = a
    else:
        threshold = a

    if _is_nan(threshold):
        return None
    return float(threshold)


def _best_split_details(X, y, min_samples_leaf=1, n_classes=None, labels_encoded=False):
    X = _as_training_array(X)
    min_samples_leaf = _validate_min_samples_leaf(min_samples_leaf)

    if labels_encoded:
        y_encoded = np.asarray(y, dtype=int).reshape(-1)
        if n_classes is None:
            n_classes = int(np.max(y_encoded)) + 1 if y_encoded.size else 0
        else:
            n_classes = int(n_classes)
    else:
        y_arr = _as_label_array(y)
        classes = _unique_sorted_labels(y_arr)
        n_classes = len(classes)
        y_encoded = _encode_labels(y_arr, classes) if n_classes else np.empty(0, dtype=int)

    if X.shape[0] != y_encoded.size:
        raise ValueError("X and y must contain the same number of samples")

    n_samples, n_features = X.shape
    if n_samples < 2 * min_samples_leaf or n_classes <= 0:
        return None, None, 0.0, True

    parent_counts = np.bincount(y_encoded, minlength=n_classes).astype(float)
    parent_impurity = _gini_from_counts(parent_counts)
    if parent_impurity <= 0.0:
        return None, None, 0.0, True

    best_feature = None
    best_threshold = None
    best_gain = 0.0
    best_nan_go_left = True

    for feature in range(n_features):
        column = X[:, feature]
        nan_mask = np.isnan(column)
        has_nan = bool(np.any(nan_mask))
        finite_mask = ~nan_mask
        finite_count = int(np.sum(finite_mask))

        if finite_count == 0:
            continue

        finite_values = column[finite_mask]
        finite_labels = y_encoded[finite_mask]
        order = np.argsort(finite_values, kind="mergesort")
        sorted_values = finite_values[order]
        sorted_labels = finite_labels[order]

        if has_nan:
            nan_counts = np.bincount(y_encoded[nan_mask], minlength=n_classes).astype(float)
        else:
            nan_counts = np.zeros(n_classes, dtype=float)

        left_finite_counts = np.zeros(n_classes, dtype=float)
        pos = 0

        while pos < finite_count:
            value = sorted_values[pos]
            end = pos + 1
            while end < finite_count and sorted_values[end] == value:
                end += 1

            left_finite_counts += np.bincount(
                sorted_labels[pos:end], minlength=n_classes
            ).astype(float)

            has_next = end < finite_count
            if has_next or has_nan:
                if has_next:
                    threshold = _threshold_between(value, sorted_values[end])
                else:
                    threshold = float(value)
                    if _is_nan(threshold):
                        threshold = None

                if threshold is not None:
                    directions = (True, False) if has_nan else (True,)
                    for nan_go_left in directions:
                        if nan_go_left:
                            left_counts = left_finite_counts + nan_counts
                        else:
                            left_counts = left_finite_counts

                        n_left = int(np.sum(left_counts))
                        n_right = n_samples - n_left
                        if n_left < min_samples_leaf or n_right < min_samples_leaf:
                            continue

                        right_counts = parent_counts - left_counts
                        left_impurity = _gini_from_counts(left_counts)
                        right_impurity = _gini_from_counts(right_counts)
                        weighted = (
                            (n_left / n_samples) * left_impurity
                            + (n_right / n_samples) * right_impurity
                        )
                        gain = float(parent_impurity - weighted)

                        if gain > best_gain + _GAIN_TOL:
                            best_feature = int(feature)
                            best_threshold = float(threshold)
                            best_gain = gain
                            best_nan_go_left = bool(nan_go_left)

            pos = end

    if best_feature is None or best_gain <= _GAIN_TOL:
        return None, None, 0.0, True

    return best_feature, best_threshold, float(best_gain), best_nan_go_left


def best_split(X, y, min_samples_leaf=1):
    feature, threshold, _gain, _nan_go_left = _best_split_details(
        X, y, min_samples_leaf=min_samples_leaf
    )
    return feature, threshold


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        if max_depth is not None:
            try:
                max_depth = int(max_depth)
            except Exception as exc:
                raise ValueError("max_depth must be None or a non-negative integer") from exc
            if max_depth < 0:
                raise ValueError("max_depth must be None or a non-negative integer")

        self.max_depth = max_depth
        self.min_samples_leaf = _validate_min_samples_leaf(min_samples_leaf)

    def fit(self, X, y):
        X = _as_training_array(X)
        y_arr = _as_label_array(y)

        if X.shape[0] != y_arr.size:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("DecisionTree cannot be fit on an empty dataset")

        classes = _unique_sorted_labels(y_arr)
        if not classes:
            raise ValueError("y must contain at least one label")

        self.classes_ = _object_array(classes)
        self.n_features_in_ = int(X.shape[1])
        self.n_classes_ = int(len(classes))

        y_encoded = _encode_labels(y_arr, classes)
        self.root_ = self._build_tree(X, y_encoded, depth=0)
        return self

    def _build_tree(self, X, y_encoded, depth):
        n_samples = int(y_encoded.size)
        counts = np.bincount(y_encoded, minlength=self.n_classes_).astype(float)
        prediction_index = int(np.argmax(counts)) if counts.size else 0
        prediction = self.classes_[prediction_index]

        node = _Node(
            prediction=prediction,
            n_samples=n_samples,
            counts=counts,
        )

        if n_samples == 0:
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            return node

        if int(np.sum(counts > 0)) <= 1:
            return node

        if n_samples < 2 * self.min_samples_leaf:
            return node

        feature, threshold, gain, nan_go_left = _best_split_details(
            X,
            y_encoded,
            min_samples_leaf=self.min_samples_leaf,
            n_classes=self.n_classes_,
            labels_encoded=True,
        )

        if feature is None or threshold is None or gain <= _GAIN_TOL:
            return node

        column = X[:, feature]
        nan_mask = np.isnan(column)
        finite_left = (~nan_mask) & (column <= threshold)
        if nan_go_left:
            left_mask = finite_left | nan_mask
        else:
            left_mask = finite_left

        n_left = int(np.sum(left_mask))
        n_right = n_samples - n_left
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return node

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.nan_go_left = bool(nan_go_left)
        node.left = self._build_tree(X[left_mask], y_encoded[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y_encoded[~left_mask], depth + 1)
        return node

    def _check_is_fitted(self):
        if not hasattr(self, "root_") or not hasattr(self, "classes_"):
            raise ValueError("DecisionTree instance is not fitted yet")

    def _prepare_X_for_prediction(self, X):
        self._check_is_fitted()
        arr = _as_prediction_array(X, self.n_features_in_)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has %d features, but DecisionTree was fitted with %d features"
                % (arr.shape[1], self.n_features_in_)
            )
        return arr

    def _leaf_for_row(self, row):
        node = self.root_
        while not node.is_leaf:
            value = row[node.feature]
            if np.isnan(value):
                go_left = node.nan_go_left
            else:
                go_left = value <= node.threshold
            node = node.left if go_left else node.right
        return node

    def predict(self, X) -> np.ndarray:
        X = self._prepare_X_for_prediction(X)
        predictions = [self._leaf_for_row(row).prediction for row in X]
        return _object_array(predictions)

    def predict_proba(self, X) -> np.ndarray:
        X = self._prepare_X_for_prediction(X)
        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)

        for i, row in enumerate(X):
            leaf = self._leaf_for_row(row)
            counts = np.asarray(leaf.counts, dtype=float)
            total = float(np.sum(counts))
            if total > 0.0:
                proba[i, :] = counts / total

        return proba