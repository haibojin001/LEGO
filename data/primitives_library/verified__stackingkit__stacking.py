import copy
import numpy as np


__all__ = ["StackingClassifier"]


def _as_integer(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    try:
        if integer != value:
            raise ValueError(f"{name} must be an integer")
    except ValueError:
        raise ValueError(f"{name} must be an integer")
    return integer


def _as_float_2d_array(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 0:
        raise ValueError("X must be a 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array-like object")
    if arr.shape[0] < 1:
        raise ValueError("X must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values")
    return arr


def _as_1d_target(y):
    arr = np.asarray(y)
    if arr.ndim == 0:
        raise ValueError("y must be a 1D array-like object")
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError("y must be a 1D array-like object")
    if arr.shape[0] < 1:
        raise ValueError("y must contain at least one label")
    return arr


def _is_nan(value):
    try:
        return bool(value != value)
    except Exception:
        return False


def _labels_equal(a, b):
    if _is_nan(a) and _is_nan(b):
        return True
    try:
        eq = a == b
    except Exception:
        return False
    try:
        return bool(eq)
    except Exception:
        return False


def _ordered_inverse_1d(values):
    labels = []
    inverse = np.empty(len(values), dtype=np.int64)

    for i, value in enumerate(values):
        found = -1
        for j, label in enumerate(labels):
            if _labels_equal(value, label):
                found = j
                break
        if found < 0:
            labels.append(value)
            found = len(labels) - 1
        inverse[i] = found

    return labels, inverse


def _kfold_indices(n, n_splits=5, shuffle=True, seed=0):
    n = _as_integer("n", n)
    n_splits = _as_integer("n_splits", n_splits)

    if n < 1:
        raise ValueError("n must be at least 1")
    if n_splits < 2:
        raise ValueError("n_folds must be at least 2")
    if n_splits > n:
        raise ValueError("n_folds cannot be greater than the number of samples")

    indices = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(indices)

    fold_sizes = np.full(n_splits, n // n_splits, dtype=np.int64)
    fold_sizes[: n % n_splits] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        test_idx = indices[start:stop].copy()
        train_idx = np.concatenate((indices[:start], indices[stop:])).astype(np.int64, copy=False)
        folds.append((train_idx, test_idx))
        start = stop

    return folds


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / denom


def _one_hot(y_codes, n_classes):
    out = np.zeros((len(y_codes), n_classes), dtype=float)
    out[np.arange(len(y_codes)), y_codes] = 1.0
    return out


def _clone_estimator(estimator):
    try:
        return copy.deepcopy(estimator)
    except Exception as exc:
        raise ValueError("base estimators must be cloneable with copy.deepcopy") from exc


def _fit_estimator(template, X, y):
    estimator = _clone_estimator(template)
    if not hasattr(estimator, "fit"):
        raise ValueError("each base estimator must provide a fit method")
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("each base estimator must provide a predict_proba method")

    fitted = estimator.fit(X, y)
    if fitted is not None and hasattr(fitted, "predict_proba"):
        estimator = fitted
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("each fitted base estimator must provide a predict_proba method")
    return estimator


def _class_index(value, n_classes):
    for i in range(n_classes):
        if _labels_equal(value, i):
            return i
    try:
        as_int = int(value)
    except Exception:
        return -1
    if 0 <= as_int < n_classes and _labels_equal(as_int, value):
        return as_int
    return -1


def _extract_estimator_classes(estimator, n_cols):
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = getattr(estimator, "classes", None)
    if classes is None:
        return None
    classes = np.asarray(classes)
    if classes.ndim != 1:
        return None
    if len(classes) != n_cols:
        return None
    return classes


def _aligned_predict_proba(estimator, X, n_classes, assumed_classes=None):
    raw = estimator.predict_proba(X)
    proba = np.asarray(raw, dtype=float)

    if proba.ndim == 0:
        raise ValueError("predict_proba must return a 1D or 2D array")
    if proba.ndim == 1:
        if proba.shape[0] != X.shape[0]:
            raise ValueError("predict_proba returned an array with the wrong number of rows")
        classes = _extract_estimator_classes(estimator, 1)
        if classes is not None:
            aligned = np.zeros((X.shape[0], n_classes), dtype=float)
            idx = _class_index(classes[0], n_classes)
            if idx < 0:
                raise ValueError("base estimator returned an unknown class label")
            aligned[:, idx] = proba
            return aligned
        if assumed_classes is not None and len(assumed_classes) == 1:
            aligned = np.zeros((X.shape[0], n_classes), dtype=float)
            idx = _class_index(assumed_classes[0], n_classes)
            if idx < 0:
                raise ValueError("base estimator returned an unknown class label")
            aligned[:, idx] = proba
            return aligned
        if n_classes == 1:
            return np.ones((X.shape[0], 1), dtype=float)
        if n_classes == 2:
            return np.column_stack((1.0 - proba, proba))
        raise ValueError("1D predict_proba output is only supported for binary classification")

    if proba.ndim != 2:
        raise ValueError("predict_proba must return a 1D or 2D array")
    if proba.shape[0] != X.shape[0]:
        raise ValueError("predict_proba returned an array with the wrong number of rows")

    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba must return only finite values")

    n_cols = proba.shape[1]
    aligned = np.zeros((X.shape[0], n_classes), dtype=float)

    classes = _extract_estimator_classes(estimator, n_cols)
    if classes is None and assumed_classes is not None and len(assumed_classes) == n_cols:
        classes = np.asarray(assumed_classes)
    if classes is None and n_cols == n_classes:
        classes = np.arange(n_classes, dtype=np.int64)

    if classes is not None:
        for j, cls in enumerate(classes):
            idx = _class_index(cls, n_classes)
            if idx < 0:
                raise ValueError("base estimator returned an unknown class label")
            aligned[:, idx] = proba[:, j]
        return aligned

    if n_classes == 2 and n_cols == 1:
        aligned[:, 0] = 1.0 - proba[:, 0]
        aligned[:, 1] = proba[:, 0]
        return aligned

    raise ValueError("could not align base estimator probabilities to the stacking classes")


class StackingClassifier:
    def __init__(self, base_estimators, meta_lr=0.1, meta_epochs=500, n_folds=5, seed=0):
        try:
            base_estimators = list(base_estimators)
        except TypeError:
            raise ValueError("base_estimators must be a non-empty iterable of estimators")
        if len(base_estimators) == 0:
            raise ValueError("base_estimators must contain at least one estimator")

        meta_lr = float(meta_lr)
        if not np.isfinite(meta_lr) or meta_lr <= 0.0:
            raise ValueError("meta_lr must be a positive finite number")

        meta_epochs = _as_integer("meta_epochs", meta_epochs)
        if meta_epochs < 0:
            raise ValueError("meta_epochs must be non-negative")

        n_folds = _as_integer("n_folds", n_folds)
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        seed = _as_integer("seed", seed)

        for estimator in base_estimators:
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict_proba"):
                raise ValueError("each base estimator must provide fit and predict_proba methods")

        self.base_estimators = base_estimators
        self.meta_lr = meta_lr
        self.meta_epochs = meta_epochs
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        X = _as_float_2d_array(X)
        y_arr = _as_1d_target(y)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        labels, y_codes = _ordered_inverse_1d(y_arr)
        n_samples = X.shape[0]
        n_classes = len(labels)
        if n_classes < 1:
            raise ValueError("y must contain at least one class")

        folds = _kfold_indices(n_samples, self.n_folds, shuffle=True, seed=self.seed)

        oof_parts = []
        for template in self.base_estimators:
            oof = np.zeros((n_samples, n_classes), dtype=float)
            for train_idx, test_idx in folds:
                train_classes = np.unique(y_codes[train_idx])
                estimator = _fit_estimator(template, X[train_idx], y_codes[train_idx])
                oof[test_idx] = _aligned_predict_proba(
                    estimator,
                    X[test_idx],
                    n_classes,
                    assumed_classes=train_classes,
                )
            oof_parts.append(oof)

        meta_X = np.concatenate(oof_parts, axis=1)
        meta_y = _one_hot(y_codes, n_classes)

        coef = np.zeros((meta_X.shape[1], n_classes), dtype=float)
        intercept = np.zeros(n_classes, dtype=float)

        if n_classes > 1:
            inv_n = 1.0 / float(n_samples)
            for _ in range(self.meta_epochs):
                probabilities = _softmax(meta_X.dot(coef) + intercept)
                error = probabilities - meta_y
                grad_coef = meta_X.T.dot(error) * inv_n
                grad_intercept = np.sum(error, axis=0) * inv_n
                coef -= self.meta_lr * grad_coef
                intercept -= self.meta_lr * grad_intercept

        self.classes_ = np.asarray(labels)
        self.n_classes_ = n_classes
        self.n_features_in_ = X.shape[1]
        self.meta_coef_ = coef
        self.meta_intercept_ = intercept
        self.oof_meta_features_ = meta_X

        self.base_estimators_ = []
        all_classes = np.arange(n_classes, dtype=np.int64)
        for template in self.base_estimators:
            fitted = _fit_estimator(template, X, y_codes)
            self.base_estimators_.append(fitted)

        self._full_assumed_classes_ = all_classes
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "base_estimators_") or not hasattr(self, "meta_coef_"):
            raise ValueError("StackingClassifier instance is not fitted yet")

    def _stack_features(self, X):
        self._check_is_fitted()
        X = _as_float_2d_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")

        parts = []
        for estimator in self.base_estimators_:
            parts.append(
                _aligned_predict_proba(
                    estimator,
                    X,
                    self.n_classes_,
                    assumed_classes=self._full_assumed_classes_,
                )
            )
        return np.concatenate(parts, axis=1)

    def predict_proba(self, X):
        meta_X = self._stack_features(X)
        if self.n_classes_ == 1:
            return np.ones((meta_X.shape[0], 1), dtype=float)
        return _softmax(meta_X.dot(self.meta_coef_) + self.meta_intercept_)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        codes = np.argmax(probabilities, axis=1)
        return self.classes_[codes]