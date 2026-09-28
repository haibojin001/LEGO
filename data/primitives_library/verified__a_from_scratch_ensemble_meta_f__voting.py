import numpy as np

__all__ = ["majority_vote", "soft_vote", "VotingClassifier"]


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


def _as_1d_classes(classes):
    try:
        classes_arr = np.asarray(classes, dtype=object)
    except Exception as exc:
        raise ValueError("classes must be array-like") from exc

    if classes_arr.ndim == 0:
        classes_arr = classes_arr.reshape(1)
    else:
        classes_arr = classes_arr.reshape(-1)

    if classes_arr.size == 0:
        raise ValueError("classes must contain at least one class")

    return classes_arr


def _as_1d_values(values, name):
    try:
        arr = np.asarray(values, dtype=object)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like") from exc

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    return arr


def _as_prediction_matrix(pred_matrix):
    try:
        if isinstance(pred_matrix, np.ndarray):
            pred = np.asarray(pred_matrix)
        else:
            pred = np.asarray(pred_matrix, dtype=object)
    except Exception as exc:
        raise ValueError("pred_matrix must have shape (n_estimators, n_samples)") from exc

    if pred.ndim != 2:
        raise ValueError("pred_matrix must have shape (n_estimators, n_samples)")
    if pred.shape[0] == 0:
        raise ValueError("pred_matrix must contain at least one estimator")
    return pred


def _build_class_lookup(ordered_classes):
    hashed = {}
    unhashable = []
    nan_index = None

    for index, cls in enumerate(ordered_classes):
        if _is_nan(cls):
            if nan_index is None:
                nan_index = index
            continue

        try:
            hash(cls)
        except Exception:
            unhashable.append((index, cls))
            continue

        try:
            if cls not in hashed:
                hashed[cls] = index
        except Exception:
            unhashable.append((index, cls))

    return hashed, unhashable, nan_index


def _lookup_class_index(value, hashed, unhashable, nan_index):
    if nan_index is not None and _is_nan(value):
        return nan_index

    try:
        return hashed[value]
    except KeyError:
        pass
    except Exception:
        pass

    for index, cls in unhashable:
        if _labels_equal(value, cls):
            return index

    return None


def _validate_estimators(estimators):
    try:
        pairs = list(estimators)
    except Exception as exc:
        raise ValueError("estimators must be an iterable of (name, estimator) pairs") from exc

    if len(pairs) == 0:
        raise ValueError("estimators must contain at least one estimator")

    normalized = []
    for item in pairs:
        try:
            name, estimator = item
        except Exception as exc:
            raise ValueError("each estimator must be a (name, estimator) pair") from exc
        normalized.append((name, estimator))

    return normalized


def _require_method(obj, method_name, estimator_name=None):
    method = getattr(obj, method_name, None)
    if not callable(method):
        prefix = "estimator"
        if estimator_name is not None:
            prefix = f"estimator {estimator_name!r}"
        raise ValueError(f"{prefix} must implement {method_name}")
    return method


def _classes_equal_sequence(a, b):
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if not _labels_equal(left, right):
            return False
    return True


def _align_proba_to_classes(proba, estimator, target_classes, estimator_name=None):
    proba_arr = np.asarray(proba, dtype=float)
    if proba_arr.ndim != 2:
        raise ValueError("predict_proba must return an array with shape (n_samples, n_classes)")

    if not hasattr(estimator, "classes_"):
        return proba_arr

    estimator_classes = _as_1d_classes(getattr(estimator, "classes_"))
    if proba_arr.shape[1] != estimator_classes.size:
        prefix = "estimator"
        if estimator_name is not None:
            prefix = f"estimator {estimator_name!r}"
        raise ValueError(
            f"{prefix} has classes_ of length {estimator_classes.size}, "
            f"but predict_proba returned {proba_arr.shape[1]} columns"
        )

    target_classes_arr = _as_1d_classes(target_classes)
    if _classes_equal_sequence(estimator_classes, target_classes_arr):
        return proba_arr

    hashed, unhashable, nan_index = _build_class_lookup(list(estimator_classes))
    aligned = np.zeros((proba_arr.shape[0], target_classes_arr.size), dtype=float)

    for target_index, cls in enumerate(target_classes_arr):
        source_index = _lookup_class_index(cls, hashed, unhashable, nan_index)
        if source_index is None:
            prefix = "estimator"
            if estimator_name is not None:
                prefix = f"estimator {estimator_name!r}"
            raise ValueError(f"{prefix} is missing class {cls!r} in classes_")
        aligned[:, target_index] = proba_arr[:, source_index]

    return aligned


def majority_vote(pred_matrix, classes) -> np.ndarray:
    pred = _as_prediction_matrix(pred_matrix)
    classes_arr = _as_1d_classes(classes)
    ordered_classes = _ordered_labels(classes_arr)

    n_estimators, n_samples = pred.shape
    n_classes = len(ordered_classes)
    hashed, unhashable, nan_index = _build_class_lookup(ordered_classes)

    winner_indices = np.zeros(n_samples, dtype=int)
    counts = np.zeros(n_classes, dtype=int)

    for sample_index in range(n_samples):
        counts.fill(0)
        for estimator_index in range(n_estimators):
            class_index = _lookup_class_index(
                pred[estimator_index, sample_index],
                hashed,
                unhashable,
                nan_index,
            )
            if class_index is None:
                raise ValueError("pred_matrix contains a label not present in classes")
            counts[class_index] += 1
        winner_indices[sample_index] = int(np.argmax(counts))

    winners = [ordered_classes[index] for index in winner_indices]
    return _array_from_labels(winners, reference=classes)


def soft_vote(proba_list, classes) -> np.ndarray:
    classes_arr = _as_1d_classes(classes)
    ordered_classes = _ordered_labels(classes_arr)
    n_classes = len(ordered_classes)

    if isinstance(proba_list, np.ndarray):
        raw = np.asarray(proba_list)
        if raw.ndim == 2:
            matrices = [raw]
        elif raw.ndim == 3:
            matrices = [raw[index] for index in range(raw.shape[0])]
        else:
            raise ValueError(
                "proba_list must contain arrays with shape (n_samples, n_classes)"
            )
    else:
        try:
            matrices = list(proba_list)
        except Exception as exc:
            raise ValueError("proba_list must be an iterable of probability arrays") from exc

    if len(matrices) == 0:
        raise ValueError("proba_list must contain at least one probability array")

    average = None
    n_samples = None

    for proba in matrices:
        try:
            arr = np.asarray(proba, dtype=float)
        except Exception as exc:
            raise ValueError(
                "each probability array must be numeric with shape (n_samples, n_classes)"
            ) from exc

        if arr.ndim != 2:
            raise ValueError(
                "each probability array must have shape (n_samples, n_classes)"
            )
        if arr.shape[1] != n_classes:
            raise ValueError(
                "each probability array must have one column for each class"
            )

        if n_samples is None:
            n_samples = arr.shape[0]
            average = np.zeros((n_samples, n_classes), dtype=float)
        elif arr.shape[0] != n_samples:
            raise ValueError("all probability arrays must have the same number of samples")

        average += arr

    average /= float(len(matrices))
    winner_indices = np.argmax(average, axis=1)
    winners = [ordered_classes[index] for index in winner_indices]
    return _array_from_labels(winners, reference=classes)


class VotingClassifier:
    def __init__(self, estimators, voting='hard'):
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        self.estimators = _validate_estimators(estimators)
        self.voting = voting

    def fit(self, X, y):
        y_arr = _as_1d_classes(y)
        ordered_classes = _ordered_labels(y_arr)
        self.classes_ = _array_from_labels(ordered_classes, reference=y)

        fitted_estimators = []
        for name, estimator in self.estimators:
            fit_method = _require_method(estimator, "fit", name)
            _require_method(estimator, "predict", name)
            if self.voting == "soft":
                _require_method(estimator, "predict_proba", name)

            fitted = fit_method(X, y)
            if fitted is None:
                fitted = estimator
            fitted_estimators.append((name, fitted))

        self.estimators_ = fitted_estimators
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "classes_") or not hasattr(self, "estimators_"):
            raise ValueError("VotingClassifier instance is not fitted yet")

        if self.voting == "hard":
            predictions = []
            n_samples = None

            for name, estimator in self.estimators_:
                predict_method = _require_method(estimator, "predict", name)
                pred = _as_1d_values(predict_method(X), "prediction")
                if n_samples is None:
                    n_samples = pred.size
                elif pred.size != n_samples:
                    raise ValueError("all estimators must predict the same number of samples")
                predictions.append(pred)

            pred_matrix = np.asarray(predictions, dtype=object)
            return majority_vote(pred_matrix, self.classes_)

        probabilities = []
        n_samples = None

        for name, estimator in self.estimators_:
            predict_proba_method = _require_method(estimator, "predict_proba", name)
            proba = _align_proba_to_classes(
                predict_proba_method(X),
                estimator,
                self.classes_,
                name,
            )
            if n_samples is None:
                n_samples = proba.shape[0]
            elif proba.shape[0] != n_samples:
                raise ValueError("all estimators must predict the same number of samples")
            probabilities.append(proba)

        return soft_vote(probabilities, self.classes_)

    def score(self, X, y) -> float:
        pred = _as_1d_values(self.predict(X), "prediction")
        y_arr = _as_1d_values(y, "y")

        if y_arr.size == 0:
            raise ValueError("y must contain at least one sample")
        if pred.size != y_arr.size:
            raise ValueError("predictions and y must contain the same number of samples")

        correct = 0
        for predicted, actual in zip(pred, y_arr):
            if _labels_equal(predicted, actual):
                correct += 1

        return float(correct) / float(y_arr.size)