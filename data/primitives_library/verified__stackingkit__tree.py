import numpy as np

__all__ = ["DecisionTree"]

_GAIN_TOL = 1e-15


class _Node:
    def __init__(self, prediction, counts, feature=None, threshold=None, left=None, right=None):
        self.prediction = prediction
        self.counts = counts
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.feature is None


def _as_float_2d_array(X, n_features=None):
    arr = np.asarray(X, dtype=float)

    if arr.ndim == 0:
        raise ValueError("X must be a 2D array-like object")

    if arr.ndim == 1:
        if n_features is None:
            arr = arr.reshape(-1, 1)
        elif n_features == 1:
            arr = arr.reshape(-1, 1)
        elif arr.shape[0] == n_features:
            arr = arr.reshape(1, -1)
        else:
            raise ValueError("X must be a 2D array-like object with the fitted number of features")

    if arr.ndim != 2:
        raise ValueError("X must be a 2D array-like object")

    return arr


def _as_int_label_array(y):
    arr = np.asarray(y)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    if arr.size == 0:
        return arr.astype(np.int64)

    if arr.dtype.kind in ("b", "i", "u"):
        return arr.astype(np.int64, copy=False)

    try:
        as_float = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError("y must contain integer class labels")

    if not np.all(np.isfinite(as_float)):
        raise ValueError("y must contain finite integer class labels")

    if not np.all(as_float == np.floor(as_float)):
        raise ValueError("y must contain integer class labels")

    return as_float.astype(np.int64)


def _gini_from_counts(counts):
    total = int(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts.astype(float) / float(total)
    return float(1.0 - np.sum(probs * probs))


def _safe_threshold_between(left_value, right_value):
    midpoint = (float(left_value) + float(right_value)) / 2.0
    if float(left_value) < midpoint < float(right_value):
        return midpoint
    return float(left_value)


def _split_gain(parent_impurity, left_counts, right_counts, n_total):
    left_n = int(np.sum(left_counts))
    right_n = int(np.sum(right_counts))

    if left_n <= 0 or right_n <= 0:
        return -np.inf

    left_impurity = _gini_from_counts(left_counts)
    right_impurity = _gini_from_counts(right_counts)

    weighted = (float(left_n) / float(n_total)) * left_impurity
    weighted += (float(right_n) / float(n_total)) * right_impurity
    return float(parent_impurity - weighted)


def _best_split(X, y_codes, n_classes):
    n_samples, n_features = X.shape
    if n_samples <= 1 or n_features == 0:
        return None

    parent_counts = np.bincount(y_codes, minlength=n_classes).astype(np.int64, copy=False)
    parent_impurity = _gini_from_counts(parent_counts)

    if parent_impurity <= 0.0:
        return None

    best_feature = None
    best_threshold = None
    best_gain = 0.0

    for feature in range(n_features):
        column = X[:, feature]
        valid_mask = ~np.isnan(column)
        n_valid = int(np.sum(valid_mask))

        if n_valid == 0:
            continue

        values = column[valid_mask]
        labels = y_codes[valid_mask]

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_labels = labels[order]

        left_counts = np.zeros(n_classes, dtype=np.int64)
        right_counts = parent_counts.copy()

        for i in range(n_valid - 1):
            cls = int(sorted_labels[i])
            left_counts[cls] += 1
            right_counts[cls] -= 1

            current_value = sorted_values[i]
            next_value = sorted_values[i + 1]

            if current_value == next_value:
                continue

            threshold = _safe_threshold_between(current_value, next_value)
            gain = _split_gain(parent_impurity, left_counts, right_counts, n_samples)

            if gain > best_gain + _GAIN_TOL:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        cls = int(sorted_labels[n_valid - 1])
        left_counts[cls] += 1
        right_counts[cls] -= 1

        if n_valid < n_samples:
            threshold = float(sorted_values[n_valid - 1])
            gain = _split_gain(parent_impurity, left_counts, right_counts, n_samples)

            if gain > best_gain + _GAIN_TOL:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    if best_feature is None or best_gain <= _GAIN_TOL:
        return None

    return best_feature, best_threshold, best_gain


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, seed=0):
        if max_depth is not None:
            if isinstance(max_depth, bool):
                raise ValueError("max_depth must be a non-negative integer or None")
            max_depth = int(max_depth)
            if max_depth < 0:
                raise ValueError("max_depth must be a non-negative integer or None")

        if isinstance(min_samples_split, bool):
            raise ValueError("min_samples_split must be an integer >= 2")
        min_samples_split = int(min_samples_split)
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = int(seed)

    def fit(self, X, y):
        X_arr = _as_float_2d_array(X)
        y_arr = _as_int_label_array(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if y_arr.shape[0] == 0:
            raise ValueError("Cannot fit a decision tree on an empty dataset")

        self.n_features_in_ = int(X_arr.shape[1])
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = int(self.classes_.shape[0])

        y_codes = np.searchsorted(self.classes_, y_arr).astype(np.int64, copy=False)

        self.root_ = self._build_tree(X_arr, y_codes, depth=0)
        self.depth_ = self._tree_depth(self.root_)
        return self

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _as_float_2d_array(X, n_features=self.n_features_in_)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")

        predictions = np.empty(X_arr.shape[0], dtype=self.classes_.dtype)
        for i, row in enumerate(X_arr):
            leaf = self._leaf_for_row(row)
            predictions[i] = leaf.prediction

        return predictions

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _as_float_2d_array(X, n_features=self.n_features_in_)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")

        proba = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)

        for i, row in enumerate(X_arr):
            leaf = self._leaf_for_row(row)
            total = float(np.sum(leaf.counts))
            if total > 0.0:
                proba[i, :] = leaf.counts.astype(float) / total

        return proba

    def _check_is_fitted(self):
        if not hasattr(self, "root_"):
            raise ValueError("This DecisionTree instance is not fitted yet")

    def _build_tree(self, X, y_codes, depth):
        counts = np.bincount(y_codes, minlength=self.n_classes_).astype(np.int64, copy=False)
        majority_code = int(np.argmax(counts))
        prediction = self.classes_[majority_code]
        node = _Node(prediction=prediction, counts=counts)

        n_samples = int(X.shape[0])

        if counts[majority_code] == n_samples:
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            return node

        if n_samples < self.min_samples_split:
            return node

        split = _best_split(X, y_codes, self.n_classes_)
        if split is None:
            return node

        feature, threshold, _gain = split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_n = int(np.sum(left_mask))
        right_n = int(np.sum(right_mask))

        if left_n == 0 or right_n == 0:
            return node

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.left = self._build_tree(X[left_mask], y_codes[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y_codes[right_mask], depth + 1)
        return node

    def _leaf_for_row(self, row):
        node = self.root_

        while not node.is_leaf:
            value = row[node.feature]
            if value <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node

    def _tree_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        left_depth = self._tree_depth(node.left)
        right_depth = self._tree_depth(node.right)
        return 1 + max(left_depth, right_depth)