import math
import operator

import numpy as np

__all__ = ["best_split_mse", "RegressionTree"]

_GAIN_TOL = 1e-15


def best_split_mse(X, target, min_samples_leaf=1):
    """
    Find the best regression-tree split under mean-squared-error impurity.

    Returns (feature, threshold, gain), where the split sends samples with
    X[:, feature] <= threshold to the left child. The gain is the reduction in
    weighted child MSE relative to parent MSE. If no positive-gain split
    satisfies min_samples_leaf, returns (None, None, 0.0).
    """
    min_samples_leaf = _validate_min_samples_leaf(min_samples_leaf)
    X_arr = _as_2d_float_array(X, "X")
    y_arr = _as_1d_float_array(target, "target")

    n_samples = X_arr.shape[0]
    if y_arr.shape[0] != n_samples:
        raise ValueError("X and target have incompatible lengths")
    _validate_finite_target(y_arr)

    if n_samples == 0 or n_samples < 2 * min_samples_leaf:
        return None, None, 0.0

    n_features = X_arr.shape[1]
    if n_features == 0:
        return None, None, 0.0

    total_sum = float(np.sum(y_arr, dtype=np.float64))
    parent_mean = total_sum / n_samples

    best_feature = None
    best_threshold = None
    best_gain = 0.0

    max_left_count = n_samples - min_samples_leaf

    for feature in range(n_features):
        column = X_arr[:, feature]

        not_nan = ~np.isnan(column)
        finite_indices = np.nonzero(not_nan)[0]
        n_not_nan = int(finite_indices.shape[0])
        if n_not_nan == 0:
            continue

        order = finite_indices[np.argsort(column[finite_indices], kind="mergesort")]
        sorted_values = column[order]
        sorted_target = y_arr[order]
        prefix_sum = np.cumsum(sorted_target, dtype=np.float64)
        nan_count = n_samples - n_not_nan

        for i in range(n_not_nan):
            left_count = i + 1
            if left_count < min_samples_leaf:
                continue
            if left_count > max_left_count:
                break

            if i < n_not_nan - 1:
                current_value = sorted_values[i]
                next_value = sorted_values[i + 1]
                if current_value == next_value:
                    continue
                threshold = _midpoint_threshold(current_value, next_value)
            else:
                if nan_count <= 0:
                    continue
                threshold = float(sorted_values[i])

            right_count = n_samples - left_count
            if right_count < min_samples_leaf:
                continue

            left_sum = float(prefix_sum[i])
            right_sum = total_sum - left_sum

            left_mean = left_sum / left_count
            right_mean = right_sum / right_count

            left_diff = left_mean - parent_mean
            right_diff = right_mean - parent_mean
            gain = (
                left_count * left_diff * left_diff
                + right_count * right_diff * right_diff
            ) / n_samples

            if gain > best_gain + _GAIN_TOL:
                best_gain = float(gain)
                best_feature = int(feature)
                best_threshold = float(threshold)

    if best_feature is None or best_gain <= _GAIN_TOL:
        return None, None, 0.0

    return best_feature, best_threshold, float(best_gain)


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.max_depth = _validate_max_depth(max_depth)
        self.min_samples_leaf = _validate_min_samples_leaf(min_samples_leaf)
        self._root = None
        self.n_features_ = None

    def fit(self, X, target):
        X_arr = _as_2d_float_array(X, "X")
        y_arr = _as_1d_float_array(target, "target")

        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and target have incompatible lengths")
        if X_arr.shape[0] == 0:
            raise ValueError("RegressionTree.fit requires at least one sample")
        _validate_finite_target(y_arr)

        self.n_features_ = int(X_arr.shape[1])
        self._root = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def predict(self, X) -> np.ndarray:
        if self._root is None:
            raise ValueError("RegressionTree instance is not fitted")

        X_arr = _as_2d_float_array_for_predict(X, self.n_features_)
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                "X has incompatible number of features: expected "
                f"{self.n_features_}, got {X_arr.shape[1]}"
            )

        out = np.empty(X_arr.shape[0], dtype=float)
        for i in range(X_arr.shape[0]):
            out[i] = self._predict_row(X_arr[i])
        return out

    def _build_tree(self, X, y, depth):
        node = _TreeNode(value=float(np.mean(y, dtype=np.float64)))

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if X.shape[0] < 2 * self.min_samples_leaf:
            return node

        feature, threshold, gain = best_split_mse(
            X, y, min_samples_leaf=self.min_samples_leaf
        )
        if feature is None or gain <= 0.0:
            return node

        mask = _split_mask(X[:, feature], threshold)
        left_count = int(np.sum(mask))
        right_count = X.shape[0] - left_count

        if (
            left_count < self.min_samples_leaf
            or right_count < self.min_samples_leaf
        ):
            return node

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_row(self, row):
        node = self._root
        while node.feature is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class _TreeNode:
    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value):
        self.value = float(value)
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def _split_mask(column, threshold) -> np.ndarray:
    return np.asarray(np.asarray(column) <= threshold, dtype=bool)


def _validate_min_samples_leaf(value):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("min_samples_leaf must be an integer") from exc
    if value < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    return int(value)


def _validate_max_depth(value):
    if value is None:
        return None
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("max_depth must be an integer or None") from exc
    if value < 0:
        raise ValueError("max_depth must be non-negative")
    return int(value)


def _as_2d_float_array(X, name):
    try:
        arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 1D or 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array-like object")
    return arr


def _as_2d_float_array_for_predict(X, n_features):
    try:
        arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be numeric") from exc

    if arr.ndim == 0:
        if n_features == 1:
            return arr.reshape(1, 1)
        raise ValueError("X must be a 1D or 2D array-like object")

    if arr.ndim == 1:
        if n_features == 0 and arr.size == 0:
            return arr.reshape(1, 0)
        if n_features == 1:
            return arr.reshape(-1, 1)
        if arr.size == n_features:
            return arr.reshape(1, -1)
        return arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")
    return arr


def _as_1d_float_array(values, name):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _validate_finite_target(y):
    if not np.all(np.isfinite(y)):
        raise ValueError("target must contain only finite values")


def _midpoint_threshold(left_value, right_value):
    left_value = float(left_value)
    right_value = float(right_value)

    if math.isfinite(left_value) and math.isfinite(right_value):
        midpoint = (left_value + right_value) / 2.0
        if not math.isfinite(midpoint) or midpoint <= left_value or midpoint >= right_value:
            midpoint = left_value + (right_value - left_value) / 2.0
        if math.isfinite(midpoint) and left_value < midpoint < right_value:
            return float(midpoint)

    return float(left_value)