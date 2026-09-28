"""Cross-validation utilities for ensemblekit.

This module implements small, dependency-light cross-validation helpers using
only NumPy and the Python standard library.
"""

import numpy as np


__all__ = ["kfold_indices", "cross_val_score"]


def _as_integer(name, value):
    """Convert an integer-like value to int, rejecting booleans and non-integers."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if integer != value:
        raise ValueError(f"{name} must be an integer")
    return integer


def kfold_indices(n, n_splits=5, shuffle=False, seed=0):
    """Return train/test indices for K-fold cross-validation.

    The samples are partitioned exactly once into test folds. Fold sizes differ
    by at most one sample; the first ``n % n_splits`` folds receive one extra
    sample.
    """
    n = _as_integer("n", n)
    n_splits = _as_integer("n_splits", n_splits)

    if n < 1:
        raise ValueError("n must be at least 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n:
        raise ValueError("n_splits cannot be greater than n")

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


def _num_samples(obj):
    try:
        return len(obj)
    except TypeError:
        arr = np.asarray(obj)
        if arr.ndim == 0:
            raise ValueError("Expected an indexable collection of samples")
        return arr.shape[0]


def _safe_index(obj, indices):
    """Index NumPy arrays and common Python sequences without external deps."""
    indices = np.asarray(indices, dtype=np.int64)

    if hasattr(obj, "iloc"):
        return obj.iloc[indices]

    try:
        return obj[indices]
    except Exception:
        pass

    if isinstance(obj, tuple):
        return tuple(obj[int(i)] for i in indices)

    return [obj[int(i)] for i in indices]


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
    """Return labels in first-seen order and integer inverse codes."""
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


def _as_1d_target(y):
    arr = np.asarray(y)
    if arr.ndim == 0:
        raise ValueError("Expected y to contain one target per sample")
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)
    return None


def _classification_codes(y, n_splits):
    """Return inverse class codes when y looks like a classification target."""
    target = _as_1d_target(y)
    if target is None:
        return None

    n = target.shape[0]
    if n == 0:
        return None

    labels, inverse = _ordered_inverse_1d(target)
    n_classes = len(labels)

    if n_classes < 1:
        return None
    if n_classes == n and n > 1:
        return None

    counts = np.bincount(inverse, minlength=n_classes)

    kind = target.dtype.kind
    if kind in "buiUSO":
        return inverse

    if kind in "fc":
        finite = True
        for value in target:
            try:
                if not np.isfinite(value):
                    finite = False
                    break
            except Exception:
                finite = False
                break
        if finite and n_classes <= max(20, n // 5) and np.max(counts) >= 2:
            return inverse

    return None


def _stratified_kfold_indices(y, n_splits=5, shuffle=True, seed=0):
    """Small NumPy-only StratifiedKFold-style splitter."""
    n = _num_samples(y)
    n_splits = _as_integer("n_splits", n_splits)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n:
        raise ValueError("n_splits cannot be greater than n")

    inverse = _classification_codes(y, n_splits)
    if inverse is None:
        return kfold_indices(n, n_splits=n_splits, shuffle=shuffle, seed=seed)

    n_classes = int(np.max(inverse)) + 1
    y_order = np.sort(inverse)

    allocation = np.zeros((n_splits, n_classes), dtype=np.int64)
    for fold in range(n_splits):
        allocation[fold] = np.bincount(y_order[fold::n_splits], minlength=n_classes)

    test_folds = np.empty(n, dtype=np.int64)
    rng = np.random.RandomState(seed)

    for class_idx in range(n_classes):
        folds_for_class = np.repeat(np.arange(n_splits, dtype=np.int64), allocation[:, class_idx])
        if shuffle:
            rng.shuffle(folds_for_class)
        test_folds[inverse == class_idx] = folds_for_class

    splits = []
    all_indices = np.arange(n, dtype=np.int64)
    for fold in range(n_splits):
        test_mask = test_folds == fold
        test_idx = all_indices[test_mask].copy()
        train_idx = all_indices[~test_mask].copy()
        splits.append((train_idx, test_idx))

    return splits


def _accuracy_score(y_true, y_pred):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape == y_pred_arr.shape:
        if y_true_arr.ndim <= 1:
            matches = y_true_arr == y_pred_arr
        else:
            axes = tuple(range(1, y_true_arr.ndim))
            matches = np.all(y_true_arr == y_pred_arr, axis=axes)
    else:
        y_true_flat = y_true_arr.ravel()
        y_pred_flat = y_pred_arr.ravel()
        if y_true_flat.shape[0] != y_pred_flat.shape[0]:
            raise ValueError("Predictions and targets have incompatible lengths")
        matches = y_true_flat == y_pred_flat

    if matches.size == 0:
        raise ValueError("Cannot compute accuracy on an empty test fold")

    return float(np.mean(matches))


def cross_val_score(make_estimator, X, y, cv=5):
    """Evaluate an estimator factory with K-fold cross-validation.

    ``make_estimator`` must be a zero-argument callable returning a fresh
    estimator. Each estimator must implement ``fit(X_train, y_train)`` and
    ``predict(X_test)``. The returned scores are per-fold accuracies.
    """
    if not callable(make_estimator):
        raise ValueError("make_estimator must be callable")

    n_x = _num_samples(X)
    n_y = _num_samples(y)
    if n_x != n_y:
        raise ValueError("X and y must contain the same number of samples")

    if isinstance(cv, (int, np.integer)) and not isinstance(cv, (bool, np.bool_)):
        cv_int = int(cv)
        if _classification_codes(y, cv_int) is not None:
            splits = _stratified_kfold_indices(y, n_splits=cv_int, shuffle=True, seed=0)
        else:
            splits = kfold_indices(n_y, n_splits=cv_int, shuffle=True, seed=0)
    else:
        splits = list(cv)

    scores = []
    for train_idx, test_idx in splits:
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        estimator = make_estimator()
        if estimator is None:
            raise ValueError("make_estimator returned None")
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise ValueError("estimator must implement fit and predict")

        X_train = _safe_index(X, train_idx)
        y_train = _safe_index(y, train_idx)
        X_test = _safe_index(X, test_idx)
        y_test = _safe_index(y, test_idx)

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        scores.append(_accuracy_score(y_test, y_pred))

    return np.asarray(scores, dtype=float)