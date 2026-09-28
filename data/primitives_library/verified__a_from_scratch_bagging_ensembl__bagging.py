import numpy as np

from baggingkit.sampling import bootstrap_sample
from baggingkit.tree import DecisionTree
from baggingkit.voting import _array_from_labels, _labels_equal, _ordered_labels, majority_vote

__all__ = ["BaggingClassifier"]


def _as_2d_array(X, *, name="X"):
    try:
        arr = np.asarray(X)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like") from exc

    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like object")
    return arr


def _as_1d_labels(y, *, name="y"):
    try:
        arr = np.asarray(y, dtype=object)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like") from exc

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 1:
        pass
    elif arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    else:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _validate_n_estimators(n_estimators):
    if isinstance(n_estimators, (bool, np.bool_)):
        raise ValueError("n_estimators must be a positive integer")
    try:
        value = int(n_estimators)
    except Exception as exc:
        raise ValueError("n_estimators must be a positive integer") from exc

    try:
        same_value = float(n_estimators) == float(value)
    except Exception:
        same_value = n_estimators == value

    if not same_value or value <= 0:
        raise ValueError("n_estimators must be a positive integer")
    return value


def _normalize_voting(voting):
    if voting not in ("hard", "soft"):
        raise ValueError("voting must be either 'hard' or 'soft'")
    return voting


def _seed_value(seed):
    if isinstance(seed, (bool, np.bool_)):
        return int(seed)
    try:
        return int(seed)
    except Exception as exc:
        raise ValueError("seed must be an integer") from exc


def _manual_oob_indices(n_samples, in_bag):
    seen = np.zeros(n_samples, dtype=bool)
    if in_bag.size:
        seen[in_bag] = True
    return np.nonzero(~seen)[0].astype(int, copy=False)


def _safe_bootstrap_indices(n_samples, seed):
    sample = np.asarray(bootstrap_sample(n_samples, seed), dtype=int).reshape(-1)
    if sample.shape[0] != n_samples:
        raise ValueError("bootstrap_sample must return exactly n indices")
    if np.any(sample < 0) or np.any(sample >= n_samples):
        raise ValueError("bootstrap_sample returned an out-of-bounds index")
    return sample


class BaggingClassifier:
    def __init__(self, n_estimators=25, max_depth=None, voting="hard", seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.voting = voting
        self.seed = seed

    def fit(self, X, y):
        n_estimators = _validate_n_estimators(self.n_estimators)
        voting = _normalize_voting(self.voting)
        seed = _seed_value(self.seed)

        X_arr = _as_2d_array(X, name="X")
        y_arr = _as_1d_labels(y, name="y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        classes_list = _ordered_labels(y_arr)
        if not classes_list:
            raise ValueError("y must contain at least one class")

        self.classes_ = _array_from_labels(classes_list, y_arr)
        self.estimators_ = []
        self.estimators_samples_ = []
        self.in_bag_indices_ = []
        self.oob_indices_ = []
        self.n_features_in_ = int(X_arr.shape[1])

        n_samples = int(X_arr.shape[0])

        for i in range(n_estimators):
            in_bag = _safe_bootstrap_indices(n_samples, seed + i)

            tree_max_depth = self.max_depth
            if tree_max_depth is None:
                tree_max_depth = n_samples

            tree = DecisionTree(max_depth=tree_max_depth)
            tree.fit(X_arr[in_bag], y_arr[in_bag])

            self.estimators_.append(tree)
            self.estimators_samples_.append(in_bag.copy())
            self.in_bag_indices_.append(in_bag.copy())
            self.oob_indices_.append(_manual_oob_indices(n_samples, in_bag))

        self.oob_score_ = self._compute_oob_score(X_arr, y_arr, voting)
        return self

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        voting = _normalize_voting(self.voting)
        X_arr = _as_2d_array(X, name="X")

        if voting == "soft":
            return self._predict_soft(X_arr)
        return self._predict_hard(X_arr)

    def score(self, X, y) -> float:
        pred = self.predict(X)
        y_arr = _as_1d_labels(y, name="y")

        if pred.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if y_arr.shape[0] == 0:
            return float("nan")

        correct = 0
        for actual, predicted in zip(y_arr, pred):
            if _labels_equal(actual, predicted):
                correct += 1
        return float(correct / y_arr.shape[0])

    def _check_is_fitted(self):
        if not hasattr(self, "classes_") or not hasattr(self, "estimators_"):
            raise ValueError("BaggingClassifier instance is not fitted yet")
        if len(self.estimators_) == 0:
            raise ValueError("BaggingClassifier instance has no fitted estimators")

    def _predict_hard(self, X_arr):
        n_samples = X_arr.shape[0]
        rows = []

        for estimator in self.estimators_:
            pred = _as_1d_labels(estimator.predict(X_arr), name="estimator prediction")
            if pred.shape[0] != n_samples:
                raise ValueError("estimator returned an invalid number of predictions")
            rows.append(pred)

        pred_matrix = np.asarray(rows, dtype=object)
        return majority_vote(pred_matrix, self.classes_)

    def _predict_soft(self, X_arr):
        n_samples = X_arr.shape[0]
        classes_arr = np.asarray(self.classes_, dtype=object).reshape(-1)
        n_classes = classes_arr.shape[0]

        if n_samples == 0:
            return _array_from_labels([], self.classes_)

        proba_sum = np.zeros((n_samples, n_classes), dtype=float)
        for estimator in self.estimators_:
            proba_sum += self._predict_proba_aligned(estimator, X_arr)

        return self._labels_from_proba(proba_sum / float(len(self.estimators_)))

    def _compute_oob_score(self, X_arr, y_arr, voting):
        if voting == "soft":
            return self._compute_oob_score_soft(X_arr, y_arr)
        return self._compute_oob_score_hard(X_arr, y_arr)

    def _compute_oob_score_hard(self, X_arr, y_arr):
        n_samples = X_arr.shape[0]
        votes = [[] for _ in range(n_samples)]

        for estimator, oob_idx in zip(self.estimators_, self.oob_indices_):
            if oob_idx.size == 0:
                continue
            pred = _as_1d_labels(estimator.predict(X_arr[oob_idx]), name="estimator prediction")
            if pred.shape[0] != oob_idx.shape[0]:
                raise ValueError("estimator returned an invalid number of OOB predictions")
            for idx, label in zip(oob_idx, pred):
                votes[int(idx)].append(label)

        correct = 0
        total = 0
        for sample_idx, sample_votes in enumerate(votes):
            if not sample_votes:
                continue
            pred_matrix = np.asarray(sample_votes, dtype=object).reshape(-1, 1)
            predicted = majority_vote(pred_matrix, self.classes_)[0]
            if _labels_equal(predicted, y_arr[sample_idx]):
                correct += 1
            total += 1

        if total == 0:
            return float("nan")
        return float(correct / total)

    def _compute_oob_score_soft(self, X_arr, y_arr):
        n_samples = X_arr.shape[0]
        n_classes = np.asarray(self.classes_, dtype=object).reshape(-1).shape[0]

        proba_sum = np.zeros((n_samples, n_classes), dtype=float)
        counts = np.zeros(n_samples, dtype=int)

        for estimator, oob_idx in zip(self.estimators_, self.oob_indices_):
            if oob_idx.size == 0:
                continue
            proba_sum[oob_idx] += self._predict_proba_aligned(estimator, X_arr[oob_idx])
            counts[oob_idx] += 1

        correct = 0
        total = 0
        for sample_idx in range(n_samples):
            if counts[sample_idx] == 0:
                continue
            mean_proba = proba_sum[sample_idx : sample_idx + 1] / float(counts[sample_idx])
            predicted = self._labels_from_proba(mean_proba)[0]
            if _labels_equal(predicted, y_arr[sample_idx]):
                correct += 1
            total += 1

        if total == 0:
            return float("nan")
        return float(correct / total)

    def _predict_proba_aligned(self, estimator, X_arr):
        n_samples = X_arr.shape[0]
        global_classes = np.asarray(self.classes_, dtype=object).reshape(-1)
        n_classes = global_classes.shape[0]

        if hasattr(estimator, "predict_proba"):
            try:
                proba = np.asarray(estimator.predict_proba(X_arr), dtype=float)

                if proba.ndim == 1:
                    if n_samples == 1 and proba.shape[0] == n_classes:
                        proba = proba.reshape(1, -1)
                    elif n_classes == 2 and proba.shape[0] == n_samples:
                        two_col = np.empty((n_samples, 2), dtype=float)
                        two_col[:, 1] = proba
                        two_col[:, 0] = 1.0 - proba
                        proba = two_col
                    else:
                        raise ValueError("invalid probability shape")

                if proba.ndim != 2 or proba.shape[0] != n_samples:
                    raise ValueError("invalid probability shape")

                tree_classes = getattr(estimator, "classes_", None)
                if tree_classes is not None:
                    tree_classes_arr = np.asarray(tree_classes, dtype=object).reshape(-1)
                    if proba.shape[1] == tree_classes_arr.shape[0]:
                        aligned = np.zeros((n_samples, n_classes), dtype=float)
                        for tree_col, tree_label in enumerate(tree_classes_arr):
                            for global_col, global_label in enumerate(global_classes):
                                if _labels_equal(tree_label, global_label):
                                    aligned[:, global_col] += proba[:, tree_col]
                                    break
                        return aligned

                if proba.shape[1] == n_classes:
                    return proba.astype(float, copy=False)
            except Exception:
                pass

        return self._one_hot_predictions(estimator, X_arr)

    def _one_hot_predictions(self, estimator, X_arr):
        n_samples = X_arr.shape[0]
        classes_arr = np.asarray(self.classes_, dtype=object).reshape(-1)
        out = np.zeros((n_samples, classes_arr.shape[0]), dtype=float)

        pred = _as_1d_labels(estimator.predict(X_arr), name="estimator prediction")
        if pred.shape[0] != n_samples:
            raise ValueError("estimator returned an invalid number of predictions")

        for row, label in enumerate(pred):
            for col, cls in enumerate(classes_arr):
                if _labels_equal(label, cls):
                    out[row, col] = 1.0
                    break
        return out

    def _labels_from_proba(self, proba):
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("probabilities must be a 2D array")

        classes_arr = np.asarray(self.classes_, dtype=object).reshape(-1)
        if classes_arr.shape[0] == 0:
            raise ValueError("classes_ must contain at least one class")
        if proba.shape[1] != classes_arr.shape[0]:
            raise ValueError("probabilities have the wrong number of columns")

        winners = []
        for row in proba:
            best_idx = 0
            best_value = row[0]
            for idx in range(1, classes_arr.shape[0]):
                if row[idx] > best_value:
                    best_value = row[idx]
                    best_idx = idx
            winners.append(classes_arr[best_idx])

        return _array_from_labels(winners, self.classes_)