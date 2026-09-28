import math
from typing import List, Optional, Tuple

import numpy as np

from forestkit.impurity import _GAIN_TOL, _as_1d_label_list, _is_nan, _labels_equal


__all__ = ["DecisionTree"]


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


def _as_training_array(X):
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


def _as_prediction_array(X, n_features):
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


def _safe_labels_equal(a, b):
    try:
        result = _labels_equal(a, b)
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        pass

    try:
        if _safe_is_nan(a) and _safe_is_nan(b):
            return True
    except Exception:
        pass

    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False


def _safe_is_nan(value):
    try:
        return bool(_is_nan(value))
    except Exception:
        try:
            return bool(np.isnan(value))
        except Exception:
            return False


def _object_array(values):
    arr = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


def _encode_labels_preserving_order(labels):
    classes = []
    encoded = []

    for label in labels:
        found = -1
        for i, cls in enumerate(classes):
            if _safe_labels_equal(label, cls):
                found = i
                break
        if found < 0:
            classes.append(label)
            found = len(classes) - 1
        encoded.append(found)

    return np.asarray(encoded, dtype=int), classes


def _gini_from_counts(counts):
    total = int(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts.astype(float) / float(total)
    return float(1.0 - np.dot(probs, probs))


def _midpoint(a, b):
    a = float(a)
    b = float(b)

    if math.isinf(a) or math.isinf(b):
        if math.isinf(a) and math.isinf(b):
            if a < b:
                return 0.0
            return None
        if math.isinf(a):
            return float(np.nextafter(b, -np.inf))
        return float(np.nextafter(a, np.inf))

    mid = a + (b - a) * 0.5
    if not (a < mid < b):
        mid = float(np.nextafter(a, b))
        if not (a < mid < b):
            mid = float(np.nextafter(b, a))
            if not (a < mid < b):
                return None
    return float(mid)


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, seed=0):
        if max_depth is not None:
            if isinstance(max_depth, bool):
                raise ValueError("max_depth must be a non-negative integer or None")
            max_depth = int(max_depth)
            if max_depth < 0:
                raise ValueError("max_depth must be a non-negative integer or None")

        if isinstance(min_samples_leaf, bool):
            raise ValueError("min_samples_leaf must be an integer >= 1")
        min_samples_leaf = int(min_samples_leaf)
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")

        if max_features is not None:
            if isinstance(max_features, bool):
                raise ValueError("max_features must be a positive integer or None")
            max_features = int(max_features)
            if max_features < 1:
                raise ValueError("max_features must be a positive integer or None")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y):
        X_arr = _as_training_array(X)
        y_list = _as_1d_label_list(y)

        if X_arr.shape[0] != len(y_list):
            raise ValueError("X and y have inconsistent lengths")
        if len(y_list) == 0:
            raise ValueError("Cannot fit a decision tree on an empty dataset")

        y_encoded, classes = _encode_labels_preserving_order(y_list)
        if len(classes) == 0:
            raise ValueError("Cannot fit a decision tree on an empty dataset")

        self.n_features_in_ = int(X_arr.shape[1])
        self.n_classes_ = int(len(classes))
        self.classes_ = _object_array(classes)
        self._classes_list = list(classes)
        self._X = X_arr
        self._y = y_encoded
        self._rng = np.random.default_rng(self.seed)

        indices = np.arange(X_arr.shape[0], dtype=int)
        self.root_, self.depth_ = self._build_tree(indices, depth=0)

        del self._X
        del self._y
        del self._rng
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise ValueError("This DecisionTree instance is not fitted yet")

        X_arr = _as_prediction_array(X, self.n_features_in_)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")

        predictions = []
        for row in X_arr:
            class_index = self._predict_one_index(row)
            predictions.append(self._classes_list[class_index])

        return _object_array(predictions)

    def _build_tree(self, indices, depth):
        y_subset = self._y[indices]
        counts = np.bincount(y_subset, minlength=self.n_classes_)
        prediction = int(np.argmax(counts))

        node = _Node(
            prediction=prediction,
            n_samples=len(indices),
            counts=counts.copy(),
        )

        if self.max_depth is not None and depth >= self.max_depth:
            return node, depth
        if int(np.max(counts)) == len(indices):
            return node, depth
        if len(indices) < 2 * self.min_samples_leaf:
            return node, depth
        if self.n_features_in_ <= 0:
            return node, depth

        split = self._best_split(indices, counts)
        if split is None:
            return node, depth

        feature, threshold, nan_go_left, left_indices, right_indices = split

        left_node, left_depth = self._build_tree(left_indices, depth + 1)
        right_node, right_depth = self._build_tree(right_indices, depth + 1)

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.nan_go_left = bool(nan_go_left)
        node.left = left_node
        node.right = right_node

        return node, max(left_depth, right_depth)

    def _feature_subset(self):
        n_features = self.n_features_in_
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features, dtype=int)

        chosen = self._rng.choice(n_features, size=self.max_features, replace=False)
        chosen = np.asarray(chosen, dtype=int)
        chosen.sort()
        return chosen

    def _best_split(self, indices, parent_counts):
        parent_impurity = _gini_from_counts(parent_counts)
        if parent_impurity <= 0.0:
            return None

        best_gain = float(_GAIN_TOL)
        best_feature = None
        best_threshold = None
        best_nan_go_left = True
        best_left_indices = None
        best_right_indices = None

        features = self._feature_subset()
        X = self._X
        y = self._y

        for feature in features:
            values = X[indices, feature]
            nan_mask = np.isnan(values)
            has_nan = bool(np.any(nan_mask))
            non_nan_values = values[~nan_mask]

            if non_nan_values.size == 0:
                continue

            unique_values = np.unique(non_nan_values)
            thresholds = []

            if has_nan:
                min_value = float(unique_values[0])
                max_value = float(unique_values[-1])
                thresholds.append(float(np.nextafter(min_value, -np.inf)))
                thresholds.append(float(np.nextafter(max_value, np.inf)))

            if unique_values.size >= 2:
                for i in range(unique_values.size - 1):
                    a = unique_values[i]
                    b = unique_values[i + 1]
                    if a == b:
                        continue
                    threshold = _midpoint(a, b)
                    if threshold is not None:
                        thresholds.append(threshold)

            if not thresholds:
                continue

            seen_thresholds = set()
            clean_thresholds = []
            for threshold in thresholds:
                if _safe_is_nan(threshold):
                    continue
                key = float(threshold)
                if key not in seen_thresholds:
                    seen_thresholds.add(key)
                    clean_thresholds.append(float(threshold))

            for threshold in clean_thresholds:
                nan_options = (False, True) if has_nan else (False,)
                for nan_go_left in nan_options:
                    left_mask = values <= threshold
                    if has_nan:
                        if nan_go_left:
                            left_mask = np.logical_or(left_mask, nan_mask)
                        else:
                            left_mask = np.logical_and(left_mask, ~nan_mask)

                    n_left = int(np.sum(left_mask))
                    n_total = len(indices)
                    n_right = n_total - n_left

                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue

                    left_indices = indices[left_mask]
                    right_indices = indices[~left_mask]

                    left_counts = np.bincount(y[left_indices], minlength=self.n_classes_)
                    right_counts = parent_counts - left_counts

                    left_impurity = _gini_from_counts(left_counts)
                    right_impurity = _gini_from_counts(right_counts)

                    weighted_impurity = (
                        (n_left / n_total) * left_impurity
                        + (n_right / n_total) * right_impurity
                    )
                    gain = parent_impurity - weighted_impurity

                    if gain > best_gain + float(_GAIN_TOL):
                        best_gain = float(gain)
                        best_feature = int(feature)
                        best_threshold = float(threshold)
                        best_nan_go_left = bool(nan_go_left)
                        best_left_indices = left_indices
                        best_right_indices = right_indices

        if best_feature is None:
            return None

        return (
            best_feature,
            best_threshold,
            best_nan_go_left,
            best_left_indices,
            best_right_indices,
        )

    def _predict_one_index(self, row):
        node = self.root_
        while not node.is_leaf:
            value = row[node.feature]
            if _safe_is_nan(value):
                go_left = node.nan_go_left
            else:
                go_left = bool(value <= node.threshold)

            node = node.left if go_left else node.right
            if node is None:
                break

        if node is None:
            return int(self.root_.prediction)
        return int(node.prediction)