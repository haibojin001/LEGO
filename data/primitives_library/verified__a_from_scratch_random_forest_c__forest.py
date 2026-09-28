import numpy as np

from forestkit.tree import DecisionTree
from forestkit.sampling import bootstrap_sample

__all__ = ["majority_vote", "RandomForestClassifier"]


def _is_nan(value):
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _labels_equal(a, b):
    if _is_nan(a) and _is_nan(b):
        return True
    try:
        result = a == b
    except Exception:
        return False
    if isinstance(result, np.ndarray):
        return bool(result.shape == () and result.item())
    try:
        return bool(result)
    except Exception:
        return False


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


def _unique_labels(labels):
    unique = []
    for label in labels:
        if not any(_labels_equal(label, existing) for existing in unique):
            unique.append(label)
    return unique


def _ordered_labels(labels):
    unique = _unique_labels(labels)
    try:
        return list(sorted(unique))
    except Exception:
        return unique


def _array_from_labels(labels, reference=None):
    if reference is not None:
        ref = np.asarray(reference)
        if ref.dtype != object:
            try:
                return np.asarray(labels, dtype=ref.dtype)
            except Exception:
                pass
    return np.asarray(labels, dtype=object)


def majority_vote(pred_matrix, classes) -> np.ndarray:
    pred = np.asarray(pred_matrix, dtype=object)

    if pred.ndim != 2:
        raise ValueError("pred_matrix must have shape (n_estimators, n_samples)")
    if pred.shape[0] == 0:
        raise ValueError("pred_matrix must contain at least one estimator")

    classes_arr = np.asarray(classes, dtype=object)
    if classes_arr.ndim == 0:
        classes_arr = classes_arr.reshape(1)
    else:
        classes_arr = classes_arr.reshape(-1)
    if classes_arr.size == 0:
        raise ValueError("classes must contain at least one class")

    ordered_classes = _ordered_labels(classes_arr)
    winners = []

    for sample_idx in range(pred.shape[1]):
        best_class = ordered_classes[0]
        best_count = -1

        for cls in ordered_classes:
            count = 0
            for estimator_idx in range(pred.shape[0]):
                if _labels_equal(pred[estimator_idx, sample_idx], cls):
                    count += 1

            if count > best_count:
                best_count = count
                best_class = cls

        winners.append(best_class)

    return _array_from_labels(winners, classes)


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=25,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        seed=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def _max_features_count(self, n_features):
        if n_features <= 0:
            raise ValueError("X must contain at least one feature")

        max_features = self.max_features

        if max_features == "sqrt":
            count = int(np.floor(np.sqrt(n_features)))
        elif max_features is None:
            count = n_features
        elif isinstance(max_features, (int, np.integer)) and not isinstance(
            max_features, bool
        ):
            count = int(max_features)
        elif isinstance(max_features, (float, np.floating)) and not isinstance(
            max_features, bool
        ):
            value = float(max_features)
            if value <= 0:
                raise ValueError("max_features must be positive")
            if value <= 1.0:
                count = int(np.floor(value * n_features))
            else:
                count = int(np.floor(value))
        else:
            raise ValueError("max_features must be 'sqrt', None, int, or float")

        if count < 1:
            raise ValueError("max_features selects zero features")
        return min(count, n_features)

    def _make_tree(self):
        try:
            return DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        except TypeError:
            try:
                return DecisionTree(
                    self.max_depth,
                    self.min_samples_leaf,
                )
            except TypeError:
                try:
                    return DecisionTree(max_depth=self.max_depth)
                except TypeError:
                    return DecisionTree()

    def _bootstrap(self, X, y, seed):
        X_arr = _as_2d_array(X, name="bootstrap X")
        y_arr = _as_1d_labels(y, name="bootstrap y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("bootstrap X and y have inconsistent lengths")
        if X_arr.shape[0] == 0:
            raise ValueError("bootstrap X and y must contain at least one sample")

        n_samples = X_arr.shape[0]

        try:
            sampled_indices = bootstrap_sample(np.arange(n_samples), seed=seed)
            sampled_indices = np.asarray(sampled_indices)
            if sampled_indices.ndim == 0:
                raise ValueError
            sampled_indices = sampled_indices.reshape(-1)
            if sampled_indices.shape[0] != n_samples:
                raise ValueError
            sampled_indices = sampled_indices.astype(int, copy=False)
            if np.any(sampled_indices < 0) or np.any(sampled_indices >= n_samples):
                raise ValueError
        except Exception:
            rng = np.random.default_rng(seed)
            sampled_indices = rng.integers(0, n_samples, size=n_samples)

        return X_arr[sampled_indices], y_arr[sampled_indices]

    def fit(self, X, y):
        X_arr = _as_2d_array(X)
        y_arr = _as_1d_labels(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        try:
            n_estimators = int(self.n_estimators)
        except Exception as exc:
            raise ValueError("n_estimators must be an integer") from exc
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

        try:
            base_seed = int(self.seed)
        except Exception as exc:
            raise ValueError("seed must be an integer") from exc

        n_features = X_arr.shape[1]
        features_per_tree = self._max_features_count(n_features)

        self.classes_ = _array_from_labels(_ordered_labels(y_arr), y_arr)
        self.estimators_ = []
        self.feature_indices_ = []
        self.n_features_in_ = n_features

        for i in range(n_estimators):
            tree_seed = base_seed + i
            X_sample, y_sample = self._bootstrap(X_arr, y_arr, tree_seed)

            rng = np.random.default_rng(tree_seed)
            if features_per_tree == n_features:
                feature_indices = np.arange(n_features, dtype=int)
            else:
                feature_indices = np.sort(
                    rng.choice(n_features, size=features_per_tree, replace=False)
                ).astype(int)

            tree = self._make_tree()
            tree.fit(X_sample[:, feature_indices], y_sample)
            tree._forest_feature_indices = feature_indices

            self.estimators_.append(tree)
            self.feature_indices_.append(feature_indices)

        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimators_") or not hasattr(self, "classes_"):
            raise ValueError("RandomForestClassifier instance is not fitted yet")
        if len(self.estimators_) == 0:
            raise ValueError("RandomForestClassifier contains no fitted estimators")

        X_arr = _as_2d_array(X)
        if hasattr(self, "n_features_in_") and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}"
            )

        predictions = []
        for i, tree in enumerate(self.estimators_):
            feature_indices = getattr(
                tree,
                "_forest_feature_indices",
                self.feature_indices_[i],
            )
            pred = np.asarray(tree.predict(X_arr[:, feature_indices]), dtype=object)
            pred = pred.reshape(-1)
            if pred.shape[0] != X_arr.shape[0]:
                raise ValueError("base estimator returned an invalid prediction length")
            predictions.append(pred)

        pred_matrix = np.vstack(predictions)
        return majority_vote(pred_matrix, self.classes_)

    def score(self, X, y) -> float:
        y_true = _as_1d_labels(y)
        y_pred = np.asarray(self.predict(X), dtype=object).reshape(-1)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if y_true.shape[0] == 0:
            raise ValueError("score is undefined for zero samples")

        correct = 0
        for actual, predicted in zip(y_true, y_pred):
            if _labels_equal(actual, predicted):
                correct += 1

        return float(correct / y_true.shape[0])