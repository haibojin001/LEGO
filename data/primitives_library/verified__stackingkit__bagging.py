"""Bootstrap aggregation of decision trees."""

import numpy as np

from stackingkit.tree import DecisionTree, _as_float_2d_array, _as_int_label_array

__all__ = ["BaggingTrees"]


def _validate_positive_int(value, name) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_n(n) -> int:
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be a non-negative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    return n


def _bootstrap_sample(n, rng) -> np.ndarray:
    n = _validate_n(n)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    return rng.integers(0, n, size=n, dtype=np.int64)


def _oob_indices(n, in_bag) -> np.ndarray:
    n = _validate_n(n)

    arr = np.asarray(in_bag)
    if arr.size == 0:
        return np.arange(n, dtype=np.int64)

    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("in_bag must be an array of integer indices")

    flat = arr.ravel()
    if np.any(flat < 0) or np.any(flat >= n):
        raise ValueError("in_bag contains indices outside range(n)")

    present = np.zeros(n, dtype=bool)
    present[flat.astype(np.intp, copy=False)] = True
    return np.nonzero(~present)[0].astype(np.int64, copy=False)


def _class_lookup(classes):
    return {int(label): i for i, label in enumerate(np.asarray(classes).reshape(-1))}


def _aligned_predict_proba(estimator, X, classes) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(X), dtype=float)

    if raw.ndim != 2:
        raise ValueError("member predict_proba must return a 2D array")
    if raw.shape[0] != X.shape[0]:
        raise ValueError("member predict_proba returned the wrong number of rows")

    classes = np.asarray(classes).reshape(-1)
    n_samples = X.shape[0]
    n_classes = classes.size

    if raw.shape[1] == n_classes:
        est_classes = getattr(estimator, "classes_", None)
        if est_classes is None:
            return raw.astype(float, copy=False)

        est_classes_arr = np.asarray(est_classes).reshape(-1)
        if est_classes_arr.size != raw.shape[1] or np.array_equal(est_classes_arr, classes):
            return raw.astype(float, copy=False)

    out = np.zeros((n_samples, n_classes), dtype=float)
    lookup = _class_lookup(classes)

    est_classes = getattr(estimator, "classes_", None)
    if est_classes is not None:
        est_classes_arr = np.asarray(est_classes).reshape(-1)
        if est_classes_arr.size == raw.shape[1]:
            for src_col, label in enumerate(est_classes_arr):
                dst_col = lookup.get(int(label))
                if dst_col is not None:
                    out[:, dst_col] += raw[:, src_col]
            return out

    for dst_col, label in enumerate(classes):
        src_col = int(label)
        if 0 <= src_col < raw.shape[1]:
            out[:, dst_col] = raw[:, src_col]

    return out


def _majority_vote(pred_matrix, classes) -> np.ndarray:
    pred = np.asarray(pred_matrix)
    if pred.ndim != 2:
        raise ValueError("pred_matrix must have shape (n_estimators, n_samples)")
    if pred.shape[0] == 0:
        raise ValueError("pred_matrix must contain at least one estimator")

    classes = np.asarray(classes).reshape(-1)
    lookup = _class_lookup(classes)

    counts = np.zeros((pred.shape[1], classes.size), dtype=np.int64)
    for est_row in pred:
        for sample_idx, label in enumerate(est_row):
            class_idx = lookup.get(int(label))
            if class_idx is not None:
                counts[sample_idx, class_idx] += 1

    return classes[np.argmax(counts, axis=1)]


class BaggingTrees:
    def __init__(self, n_estimators=10, max_depth=None, seed=0):
        self.n_estimators = _validate_positive_int(n_estimators, "n_estimators")
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        X_arr = _as_float_2d_array(X)
        y_arr = _as_int_label_array(y)

        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        classes = np.unique(y_arr)
        if classes.size == 0:
            raise ValueError("y must contain at least one class")

        rng = np.random.default_rng(self.seed)
        estimators = []
        bootstrap_indices = []
        oob = []

        n_samples = X_arr.shape[0]
        for _ in range(self.n_estimators):
            indices = _bootstrap_sample(n_samples, rng)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_arr[indices], y_arr[indices])

            estimators.append(tree)
            bootstrap_indices.append(indices)
            oob.append(_oob_indices(n_samples, indices))

        self.estimators_ = estimators
        self.bootstrap_indices_ = bootstrap_indices
        self.oob_indices_ = oob
        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "estimators_"):
            raise ValueError("This BaggingTrees instance is not fitted yet")

    def _validate_X_predict(self, X):
        self._check_is_fitted()
        X_arr = _as_float_2d_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        return X_arr

    def predict(self, X) -> np.ndarray:
        X_arr = self._validate_X_predict(X)
        pred_matrix = np.asarray([est.predict(X_arr) for est in self.estimators_])
        return _majority_vote(pred_matrix, self.classes_)

    def predict_proba(self, X) -> np.ndarray:
        X_arr = self._validate_X_predict(X)

        total = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)
        for estimator in self.estimators_:
            total += _aligned_predict_proba(estimator, X_arr, self.classes_)

        return total / float(len(self.estimators_))