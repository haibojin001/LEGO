import math
from functools import cmp_to_key
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from treekit2.impurity import _NAN_SENTINEL, _is_nan, _labels_equal


__all__ = ["Node", "DecisionTreeClassifier"]


class Node:
    def __init__(self, *args, prediction=None, feature=None, threshold=None, left=None, right=None):
        if args:
            if len(args) == 1:
                prediction = args[0]
            elif len(args) == 4:
                feature, threshold, left, right = args
            else:
                raise TypeError("Node accepts either Node(prediction) or Node(feature, threshold, left, right)")

        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Node(prediction={self.prediction!r})"
        return (
            f"Node(feature={self.feature!r}, threshold={self.threshold!r}, "
            f"left={self.left!r}, right={self.right!r})"
        )


class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        criterion = str(criterion).lower()
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        if max_depth is not None:
            if isinstance(max_depth, bool):
                raise ValueError("max_depth must be a non-negative integer or None")
            max_depth = int(max_depth)
            if max_depth < 0:
                raise ValueError("max_depth must be a non-negative integer or None")

        if isinstance(min_samples_split, bool) or int(min_samples_split) < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if isinstance(min_samples_leaf, bool) or int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, X, y):
        X_arr = _as_float_2d_array(X)
        y_list = _as_label_list(y)

        if X_arr.shape[0] != len(y_list):
            raise ValueError("X and y have inconsistent lengths")
        if len(y_list) == 0:
            raise ValueError("Cannot fit a decision tree on an empty dataset")

        self.n_features_in_ = int(X_arr.shape[1])
        self.classes_ = _classes_array(y_list)
        self.root_, self.depth_ = self._build_tree(X_arr, y_list, depth=0)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise ValueError("This DecisionTreeClassifier instance is not fitted yet")

        X_arr = _as_float_2d_array(X, n_features=getattr(self, "n_features_in_", None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")

        predictions = [self._predict_one(row) for row in X_arr]
        out = np.empty(len(predictions), dtype=object)
        out[:] = predictions
        return out

    def score(self, X, y) -> float:
        y_true = _as_label_list(y)
        y_pred = self.predict(X)

        if len(y_true) != len(y_pred):
            raise ValueError("X and y have inconsistent lengths")
        if len(y_true) == 0:
            raise ValueError("Cannot score on an empty dataset")

        correct = 0
        for pred, true in zip(y_pred, y_true):
            if _labels_same(pred, true):
                correct += 1
        return correct / len(y_true)

    def _build_tree(self, X: np.ndarray, y: List[Any], depth: int) -> Tuple[Node, int]:
        majority = _majority_label(y)

        if _is_pure(y):
            return Node(prediction=y[0]), depth

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=majority), depth

        if len(y) < self.min_samples_split:
            return Node(prediction=majority), depth

        if X.shape[1] == 0:
            return Node(prediction=majority), depth

        split = self._best_split(X, y)
        if split is None:
            return Node(prediction=majority), depth

        feature, threshold, left_mask = split
        right_mask = ~left_mask

        X_left = X[left_mask]
        X_right = X[right_mask]
        y_left = [label for label, keep in zip(y, left_mask) if bool(keep)]
        y_right = [label for label, keep in zip(y, right_mask) if bool(keep)]

        left_node, left_depth = self._build_tree(X_left, y_left, depth + 1)
        right_node, right_depth = self._build_tree(X_right, y_right, depth + 1)

        node = Node(
            prediction=majority,
            feature=int(feature),
            threshold=float(threshold),
            left=left_node,
            right=right_node,
        )
        return node, max(left_depth, right_depth)

    def _best_split(self, X: np.ndarray, y: List[Any]):
        n_samples, n_features = X.shape
        parent_counts = _label_counts(y)
        parent_impurity = _impurity_from_counts(parent_counts, self.criterion)

        best_gain = -math.inf
        best_feature = None
        best_threshold = None
        best_mask = None

        for feature in range(n_features):
            values = X[:, feature]
            thresholds = _candidate_thresholds(values)

            for threshold in thresholds:
                left_mask = values <= threshold
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = [label for label, keep in zip(y, left_mask) if bool(keep)]
                y_right = [label for label, keep in zip(y, left_mask) if not bool(keep)]

                left_impurity = _impurity_from_counts(_label_counts(y_left), self.criterion)
                right_impurity = _impurity_from_counts(_label_counts(y_right), self.criterion)

                weighted = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = parent_impurity - weighted

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_mask = left_mask.copy()

        if best_feature is None:
            return None

        if best_gain < -1e-12:
            return None

        return best_feature, best_threshold, best_mask

    def _predict_one(self, row: np.ndarray):
        node = self.root_
        while not node.is_leaf:
            value = row[node.feature]
            if value <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction


def _as_float_2d_array(X, n_features: Optional[int] = None) -> np.ndarray:
    try:
        arr = np.asarray(X, dtype=float)
    except Exception as exc:
        raise ValueError("X must be convertible to a 2D numeric array") from exc

    if arr.ndim == 0:
        raise ValueError("X must be a 2D array-like object")

    if arr.ndim == 1:
        if n_features is None or n_features == 1:
            arr = arr.reshape(-1, 1)
        elif arr.size == n_features:
            arr = arr.reshape(1, -1)
        else:
            raise ValueError("1D X cannot be reshaped to the fitted number of features")
    elif arr.ndim != 2:
        raise ValueError("X must be a 2D array-like object")

    return arr


def _as_label_list(y) -> List[Any]:
    if isinstance(y, np.ndarray):
        if y.ndim == 0:
            raise ValueError("y must be a 1D array-like object")
        return y.ravel().tolist()

    try:
        return list(y)
    except TypeError as exc:
        raise ValueError("y must be a 1D array-like object") from exc


def _labels_same(a: Any, b: Any) -> bool:
    try:
        return bool(_labels_equal(a, b))
    except Exception:
        try:
            return bool(a == b)
        except Exception:
            return False


def _label_key(label: Any):
    try:
        if _is_nan(label):
            return ("nan", _NAN_SENTINEL)
    except Exception:
        pass

    try:
        hash(label)
        return ("hash", label)
    except Exception:
        return ("repr", type(label).__name__, repr(label))


def _label_counts(labels: Iterable[Any]) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for label in labels:
        key = _label_key(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _label_compare(a: Any, b: Any) -> int:
    if _labels_same(a, b):
        return 0

    a_nan = False
    b_nan = False
    try:
        a_nan = bool(_is_nan(a))
    except Exception:
        a_nan = False
    try:
        b_nan = bool(_is_nan(b))
    except Exception:
        b_nan = False

    if a_nan and not b_nan:
        return 1
    if b_nan and not a_nan:
        return -1

    try:
        if bool(a < b):
            return -1
        if bool(b < a):
            return 1
    except Exception:
        pass

    ka = (type(a).__name__, repr(a))
    kb = (type(b).__name__, repr(b))
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


def _sort_labels(labels: Iterable[Any]) -> List[Any]:
    return sorted(list(labels), key=cmp_to_key(_label_compare))


def _unique_labels(labels: Iterable[Any]) -> List[Any]:
    unique: List[Any] = []
    for label in labels:
        if not any(_labels_same(label, seen) for seen in unique):
            unique.append(label)
    return unique


def _classes_array(labels: Iterable[Any]) -> np.ndarray:
    classes = _sort_labels(_unique_labels(labels))
    arr = np.empty(len(classes), dtype=object)
    arr[:] = classes
    return arr


def _majority_label(labels: List[Any]) -> Any:
    if not labels:
        raise ValueError("Cannot choose a majority label from an empty list")

    counts: Dict[Any, int] = {}
    representatives: Dict[Any, Any] = {}

    for label in labels:
        key = _label_key(label)
        counts[key] = counts.get(key, 0) + 1
        if key not in representatives:
            representatives[key] = label

    max_count = max(counts.values())
    tied = [representatives[key] for key, count in counts.items() if count == max_count]
    return _sort_labels(tied)[0]


def _is_pure(labels: List[Any]) -> bool:
    if len(labels) <= 1:
        return True
    first = labels[0]
    return all(_labels_same(first, label) for label in labels[1:])


def _impurity_from_counts(counts: Dict[Any, int], criterion: str) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    if criterion == "gini":
        impurity = 1.0
        for count in counts.values():
            p = count / total
            impurity -= p * p
        return impurity

    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _candidate_thresholds(values: np.ndarray) -> List[float]:
    if values.size == 0:
        return []

    non_nan = values[~np.isnan(values)]
    if non_nan.size <= 1:
        return []

    unique = np.unique(non_nan)
    unique.sort()

    if unique.size <= 1:
        return []

    thresholds: List[float] = []
    for i in range(unique.size - 1):
        low = float(unique[i])
        high = float(unique[i + 1])
        if low == high:
            continue
        thresholds.append(_midpoint(low, high))

    return thresholds


def _midpoint(low: float, high: float) -> float:
    if math.isfinite(low) and math.isfinite(high):
        return low + (high - low) / 2.0
    if low == -math.inf and math.isfinite(high):
        return -math.inf
    if math.isfinite(low) and high == math.inf:
        return low
    return (low + high) / 2.0