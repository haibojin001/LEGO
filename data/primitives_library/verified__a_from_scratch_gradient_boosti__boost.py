import inspect
import math
import operator

import numpy as np

from gbkit.regtree import (
    RegressionTree,
    _as_1d_float_array,
    _as_2d_float_array,
    _as_2d_float_array_for_predict,
    _validate_max_depth,
)
from gbkit.loss import neg_gradient, sigmoid

__all__ = ["GradientBoostingClassifier"]


def _validate_nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(result)


def _validate_seed(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    try:
        return int(operator.index(value))
    except Exception as exc:
        raise ValueError("seed must be an integer") from exc


def _validate_learning_rate(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("learning_rate must be a positive finite number")
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError("learning_rate must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("learning_rate must be a positive finite number")
    return result


def _validate_binary_target(y):
    arr = _as_1d_float_array(y, "y")
    if arr.shape[0] == 0:
        raise ValueError("y must contain at least one sample")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("y must contain only 0/1 labels")
    return arr


def _log_odds_from_rate(rate):
    rate = float(rate)
    if rate <= 0.0:
        return -math.inf
    if rate >= 1.0:
        return math.inf
    return math.log(rate / (1.0 - rate))


def _make_regression_tree(max_depth, seed):
    try:
        sig = inspect.signature(RegressionTree)
        params = sig.parameters
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        kwargs = {}
        if "max_depth" in params or has_var_keyword:
            kwargs["max_depth"] = max_depth
        if "seed" in params or has_var_keyword:
            kwargs["seed"] = seed
        elif "random_state" in params:
            kwargs["random_state"] = seed

        try:
            return RegressionTree(**kwargs)
        except TypeError:
            pass
    except (TypeError, ValueError):
        pass

    try:
        return RegressionTree(max_depth=max_depth)
    except TypeError:
        return RegressionTree(max_depth)


def _as_tree_prediction(values, n_samples):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != n_samples:
        raise ValueError("tree prediction length does not match number of samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError("tree predictions must contain only finite values")
    return arr


class GradientBoostingClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, seed=0):
        self.n_estimators = _validate_nonnegative_int(n_estimators, "n_estimators")
        self.learning_rate = _validate_learning_rate(learning_rate)

        if isinstance(max_depth, (bool, np.bool_)):
            raise ValueError("max_depth must be a valid tree depth")
        validated_depth = _validate_max_depth(max_depth)
        self.max_depth = max_depth if validated_depth is None else validated_depth

        self.seed = _validate_seed(seed)

    def fit(self, X, y):
        X_arr = _as_2d_float_array(X, "X")
        y_arr = _validate_binary_target(y)

        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.n_features_in_ = int(X_arr.shape[1])
        self.init_ = _log_odds_from_rate(float(np.mean(y_arr)))
        self.trees_ = []

        raw = np.full(X_arr.shape[0], self.init_, dtype=float)

        for i in range(self.n_estimators):
            residual = np.asarray(neg_gradient(y_arr, raw), dtype=float).reshape(-1)
            if residual.shape[0] != y_arr.shape[0]:
                raise ValueError("negative gradient length does not match y")
            if not np.all(np.isfinite(residual)):
                raise ValueError("negative gradient must contain only finite values")

            tree = _make_regression_tree(self.max_depth, self.seed + i)
            tree.fit(X_arr, residual)

            update = _as_tree_prediction(tree.predict(X_arr), X_arr.shape[0])
            raw = raw + self.learning_rate * update

            self.trees_.append(tree)

        return self

    def decision_function(self, X):
        if not hasattr(self, "init_") or not hasattr(self, "trees_"):
            raise ValueError("GradientBoostingClassifier is not fitted")

        X_arr = _as_2d_float_array_for_predict(X, self.n_features_in_)
        raw = np.full(X_arr.shape[0], self.init_, dtype=float)

        for tree in self.trees_:
            raw = raw + self.learning_rate * _as_tree_prediction(
                tree.predict(X_arr), X_arr.shape[0]
            )

        return raw

    def predict_proba(self, X):
        proba = np.asarray(sigmoid(self.decision_function(X)), dtype=float).reshape(-1)
        return proba

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        y_arr = _validate_binary_target(y)
        pred = self.predict(X)

        if pred.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        return float(np.mean(pred == y_arr.astype(int)))