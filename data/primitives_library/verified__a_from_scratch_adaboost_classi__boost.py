import math
import numpy as np

from adakit.stump import DecisionStump, _labels_equal


__all__ = ["weighted_vote", "AdaBoostClassifier"]


def _is_nan(value):
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def _as_1d_object_array(values, name):
    try:
        arr = np.asarray(values, dtype=object)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like") from exc

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    return arr


def _as_1d_float_array(values, name):
    try:
        arr = np.asarray(values, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like with numeric values") from exc

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_2d_float_array(X, name="X"):
    try:
        arr = np.asarray(X, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like with numeric values") from exc

    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 2-dimensional array")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_2d_float_array_for_predict(X, n_features):
    arr = np.asarray(X, dtype=float)

    if arr.ndim == 0:
        raise ValueError("X must be a 2-dimensional array")
    if arr.ndim == 1:
        if n_features == 1:
            arr = arr.reshape(-1, 1)
        elif arr.size == n_features:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")

    if arr.shape[1] != n_features:
        raise ValueError(
            f"X has {arr.shape[1]} features, but this AdaBoostClassifier "
            f"was fitted with {n_features} features"
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values")
    return arr


def _unique_classes(y):
    raw = np.asarray(y)

    try:
        unique = np.unique(raw)
        if unique.ndim != 1:
            unique = unique.reshape(-1)
        if unique.size:
            return unique
    except Exception:
        pass

    y_obj = _as_1d_object_array(y, "y")
    unique = []
    for label in y_obj:
        if not any(_labels_equal(label, existing) for existing in unique):
            unique.append(label)

    try:
        unique = sorted(unique)
    except Exception:
        pass

    return np.asarray(unique, dtype=object)


def _class_index(label, classes):
    for index, cls in enumerate(classes):
        if _labels_equal(label, cls):
            return index
    return None


def _prediction_misses(y_true, y_pred):
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    missed = np.zeros(y_true.shape[0], dtype=bool)
    for i in range(y_true.shape[0]):
        missed[i] = not _labels_equal(y_true[i], y_pred[i])
    return missed


def _weighted_error(y_true, y_pred, sample_weight):
    missed = _prediction_misses(y_true, y_pred)
    total = float(np.sum(sample_weight))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("sample weights must sum to a positive finite value")
    return float(np.sum(sample_weight[missed]) / total)


def _samme_alpha(error, n_classes):
    if n_classes <= 1:
        return 1.0
    return float(math.log((1.0 - error) / error) + math.log(n_classes - 1.0))


def _perfect_alpha(n_classes, previous_alphas=None):
    eps = np.finfo(float).eps
    base = math.log((1.0 - eps) / eps)
    if n_classes > 1:
        base += math.log(n_classes - 1.0)

    if previous_alphas is not None and len(previous_alphas):
        previous_total = float(np.sum(np.abs(np.asarray(previous_alphas, dtype=float))))
        if math.isfinite(previous_total):
            base = max(base, previous_total + 1.0)

    return float(base)


def _make_stump(seed):
    try:
        return DecisionStump(seed=seed)
    except TypeError:
        pass

    try:
        return DecisionStump(random_state=seed)
    except TypeError:
        pass

    return DecisionStump()


def _fit_stump(stump, X, y, sample_weight, classes):
    try:
        return stump.fit(X, y, sample_weight, classes)
    except TypeError as first_exc:
        try:
            return stump.fit(X, y, sample_weight=sample_weight, classes=classes)
        except TypeError:
            try:
                return stump.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                try:
                    return stump.fit(X, y, sample_weight)
                except TypeError:
                    try:
                        return stump.fit(X, y)
                    except TypeError:
                        raise first_exc


def _predict_stump(stump, X):
    pred = stump.predict(X)
    pred = np.asarray(pred, dtype=object)
    if pred.ndim == 0:
        pred = pred.reshape(1)
    else:
        pred = pred.reshape(-1)
    return pred


def weighted_vote(pred_matrix, alphas, classes) -> np.ndarray:
    try:
        pred = np.asarray(pred_matrix)
    except Exception as exc:
        raise ValueError("pred_matrix must have shape (n_estimators, n_samples)") from exc

    if pred.ndim != 2:
        raise ValueError("pred_matrix must have shape (n_estimators, n_samples)")

    alpha_arr = _as_1d_float_array(alphas, "alphas")
    classes_arr = _as_1d_object_array(classes, "classes")

    if classes_arr.size == 0:
        raise ValueError("classes must contain at least one class")
    if pred.shape[0] == 0:
        raise ValueError("pred_matrix must contain at least one estimator")
    if alpha_arr.shape[0] != pred.shape[0]:
        raise ValueError("alphas length must match pred_matrix.shape[0]")

    n_estimators, n_samples = pred.shape
    n_classes = classes_arr.shape[0]
    scores = np.zeros((n_samples, n_classes), dtype=float)

    for estimator_index in range(n_estimators):
        alpha = float(alpha_arr[estimator_index])
        for sample_index in range(n_samples):
            class_index = _class_index(pred[estimator_index, sample_index], classes_arr)
            if class_index is None:
                raise ValueError("pred_matrix contains a label not present in classes")
            scores[sample_index, class_index] += alpha

    winner_indices = np.argmax(scores, axis=1)
    return np.asarray(classes_arr[winner_indices])


class AdaBoostClassifier:
    def __init__(self, n_estimators=50, seed=0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y):
        X_arr = _as_2d_float_array(X, "X")
        y_arr = _as_1d_object_array(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        try:
            n_estimators = int(self.n_estimators)
        except Exception as exc:
            raise ValueError("n_estimators must be a positive integer") from exc
        if n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")

        classes = _unique_classes(y_arr)
        if classes.size == 0:
            raise ValueError("y must contain at least one class")

        self.classes_ = classes
        self.estimators_ = []
        self.estimator_weights_ = []
        self.n_features_in_ = X_arr.shape[1]

        n_samples = X_arr.shape[0]
        n_classes = int(classes.shape[0])

        counts = np.zeros(n_classes, dtype=float)
        for label in y_arr:
            idx = _class_index(label, classes)
            if idx is not None:
                counts[idx] += 1.0
        self.class_counts_ = counts
        self.majority_class_ = classes[int(np.argmax(counts))]

        if n_classes == 1:
            return self

        sample_weight = np.ones(n_samples, dtype=float) / float(n_samples)
        threshold = 1.0 - 1.0 / float(n_classes)

        for m in range(n_estimators):
            stump = _make_stump(int(self.seed) + m if self.seed is not None else m)
            _fit_stump(stump, X_arr, y_arr, sample_weight.copy(), self.classes_)

            pred = _predict_stump(stump, X_arr)
            if pred.shape[0] != n_samples:
                raise ValueError("weak learner returned the wrong number of predictions")

            err = _weighted_error(y_arr, pred, sample_weight)

            if err <= 0.0:
                alpha = _perfect_alpha(n_classes, self.estimator_weights_)
                self.estimators_.append(stump)
                self.estimator_weights_.append(alpha)
                break

            if err >= threshold:
                break

            alpha = _samme_alpha(err, n_classes)
            if not math.isfinite(alpha):
                break

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)

            missed = _prediction_misses(y_arr, pred).astype(float)
            sample_weight *= np.exp(alpha * missed)
            weight_sum = float(np.sum(sample_weight))
            if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                sample_weight = np.ones(n_samples, dtype=float) / float(n_samples)
            else:
                sample_weight /= weight_sum

        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise ValueError("AdaBoostClassifier is not fitted")

        X_arr = _as_2d_float_array_for_predict(X, self.n_features_in_)

        if len(self.estimators_) == 0:
            return np.asarray([self.majority_class_] * X_arr.shape[0])

        predictions = []
        for estimator in self.estimators_:
            pred = _predict_stump(estimator, X_arr)
            if pred.shape[0] != X_arr.shape[0]:
                raise ValueError("weak learner returned the wrong number of predictions")
            predictions.append(pred)

        pred_matrix = np.vstack(predictions)
        return weighted_vote(pred_matrix, self.estimator_weights_, self.classes_)

    def score(self, X, y) -> float:
        y_true = _as_1d_object_array(y, "y")
        y_pred = np.asarray(self.predict(X), dtype=object).reshape(-1)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if y_true.shape[0] == 0:
            raise ValueError("y must contain at least one sample")

        correct = 0
        for true_label, pred_label in zip(y_true, y_pred):
            if _labels_equal(true_label, pred_label):
                correct += 1

        return float(correct / y_true.shape[0])