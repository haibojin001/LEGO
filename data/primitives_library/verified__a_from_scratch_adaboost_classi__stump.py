import math
import numpy as np

__all__ = ["weighted_gini", "DecisionStump"]


def weighted_gini(y, w, classes) -> float:
    y_arr = _as_1d_object_array(y, "y")
    w_arr = _as_1d_float_array(w, "w")

    if y_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("y and w have incompatible lengths")

    _validate_sample_weight_values(w_arr)

    if y_arr.shape[0] == 0:
        return 0.0

    classes_list = _classes_to_list(classes, require_non_empty=False)
    encoded = _encode_labels(y_arr, classes_list)
    counts = np.bincount(encoded, weights=w_arr, minlength=len(classes_list)).astype(float)
    return float(_gini_from_counts(counts))


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_class = None
        self.right_class = None
        self.classes_ = None
        self.n_features_in_ = None
        self._is_fitted = False

    def fit(self, X, y, sample_weight, classes):
        X_arr = _as_2d_float_array(X, "X")
        y_arr = _as_1d_object_array(y, "y")

        n_samples, n_features = X_arr.shape
        if y_arr.shape[0] != n_samples:
            raise ValueError("X and y have incompatible lengths")
        if n_samples == 0:
            raise ValueError("X and y must contain at least one sample")

        if sample_weight is None:
            w_arr = np.ones(n_samples, dtype=float)
        else:
            w_arr = _as_1d_float_array(sample_weight, "sample_weight")
            if w_arr.shape[0] != n_samples:
                raise ValueError("X and sample_weight have incompatible lengths")

        _validate_sample_weight_values(w_arr)

        classes_list = _classes_to_list(classes, require_non_empty=True)
        encoded_y = _encode_labels(y_arr, classes_list)
        n_classes = len(classes_list)

        parent_counts = np.bincount(
            encoded_y, weights=w_arr, minlength=n_classes
        ).astype(float)
        parent_class = _majority_class_from_counts(parent_counts, classes_list)
        total_weight = float(np.sum(w_arr))

        self.feature = None
        self.threshold = None
        self.left_class = parent_class
        self.right_class = parent_class
        self.classes_ = np.asarray(classes_list, dtype=object)
        self.n_features_in_ = n_features
        self._is_fitted = True

        if n_samples < 2:
            return self

        parent_gini = _gini_from_counts(parent_counts)
        if parent_gini <= 0.0:
            return self

        best_feature = None
        best_threshold = None
        best_score = math.inf
        best_left_counts = None
        best_right_counts = None

        for feature in range(n_features):
            column = X_arr[:, feature]
            not_nan_mask = ~np.isnan(column)
            finite_indices = np.nonzero(not_nan_mask)[0]
            nan_count = n_samples - finite_indices.shape[0]

            if finite_indices.shape[0] == 0:
                continue

            order = finite_indices[np.argsort(column[finite_indices], kind="mergesort")]
            sorted_values = column[order]
            sorted_labels = encoded_y[order]
            sorted_weights = w_arr[order]

            left_counts = np.zeros(n_classes, dtype=float)
            right_counts = parent_counts.copy()

            left_samples = 0
            right_samples = n_samples

            pos = 0
            m = order.shape[0]
            while pos < m:
                value = sorted_values[pos]
                end = pos + 1
                while end < m and sorted_values[end] == value:
                    end += 1

                group_labels = sorted_labels[pos:end]
                group_weights = sorted_weights[pos:end]
                group_counts = np.bincount(
                    group_labels, weights=group_weights, minlength=n_classes
                ).astype(float)

                left_counts += group_counts
                right_counts -= group_counts

                group_size = end - pos
                left_samples += group_size
                right_samples -= group_size

                threshold = None
                if end < m:
                    next_value = sorted_values[end]
                    threshold = _threshold_between(value, next_value)
                elif nan_count > 0:
                    threshold = float(value)

                if threshold is not None and left_samples > 0 and right_samples > 0:
                    left_weight = float(np.sum(left_counts))
                    right_weight = float(np.sum(right_counts))

                    if total_weight > 0.0:
                        score = (
                            left_weight * _gini_from_counts(left_counts)
                            + right_weight * _gini_from_counts(right_counts)
                        ) / total_weight
                    else:
                        score = 0.0

                    if score < best_score:
                        best_score = score
                        best_feature = feature
                        best_threshold = threshold
                        best_left_counts = left_counts.copy()
                        best_right_counts = right_counts.copy()

                pos = end

        if best_feature is not None:
            self.feature = int(best_feature)
            self.threshold = float(best_threshold)
            self.left_class = _majority_class_from_counts(best_left_counts, classes_list)
            self.right_class = _majority_class_from_counts(best_right_counts, classes_list)

        return self

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("DecisionStump instance is not fitted yet")

        X_arr = _as_2d_float_array_for_predict(X, self.n_features_in_)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible number of features")

        n_samples = X_arr.shape[0]
        out = np.empty(n_samples, dtype=object)

        if self.feature is None:
            out[:] = self.left_class
            return out

        mask = X_arr[:, self.feature] <= self.threshold
        out[mask] = self.left_class
        out[~mask] = self.right_class
        return out


def _as_1d_object_array(values, name):
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _as_1d_float_array(values, name):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _as_2d_float_array(values, name):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 1D or 2D array-like object")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array-like object")
    return arr


def _as_2d_float_array_for_predict(values, n_features_in):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric values") from exc

    if arr.ndim == 0:
        raise ValueError("X must be a 1D or 2D array-like object")
    if arr.ndim == 1:
        if n_features_in is not None and n_features_in != 1 and arr.shape[0] == n_features_in:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object")
    return arr


def _classes_to_list(classes, require_non_empty):
    arr = np.asarray(classes, dtype=object)
    if arr.ndim == 0:
        result = arr.reshape(1).tolist()
    else:
        result = arr.reshape(-1).tolist()

    if require_non_empty and len(result) == 0:
        raise ValueError("classes must contain at least one class")
    return result


def _validate_sample_weight_values(w):
    if np.any(~np.isfinite(w)):
        raise ValueError("sample weights must be finite")
    if np.any(w < 0):
        raise ValueError("sample weights must be non-negative")


def _encode_labels(y, classes_list):
    if len(y) == 0:
        return np.empty(0, dtype=int)
    if len(classes_list) == 0:
        raise ValueError("classes must contain all labels in y")

    encoded = np.empty(len(y), dtype=int)
    for i, label in enumerate(y):
        found = False
        for j, cls in enumerate(classes_list):
            if _labels_equal(label, cls):
                encoded[i] = j
                found = True
                break
        if not found:
            raise ValueError("classes must contain all labels in y")
    return encoded


def _labels_equal(a, b):
    try:
        eq = a == b
        if isinstance(eq, np.ndarray):
            if eq.shape == ():
                return bool(eq.item())
            return bool(np.all(eq))
        if bool(eq):
            return True
    except Exception:
        pass

    try:
        return bool(np.isnan(a) and np.isnan(b))
    except Exception:
        return False


def _gini_from_counts(counts):
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0

    probs = counts / total
    gini = 1.0 - float(np.dot(probs, probs))

    if gini < 0.0 and gini > -1e-12:
        return 0.0
    return gini


def _majority_class_from_counts(counts, classes_list):
    if len(classes_list) == 0:
        return None

    best_index = 0
    best_weight = float(counts[0]) if len(counts) else 0.0

    for i in range(1, len(classes_list)):
        weight = float(counts[i])
        if weight > best_weight:
            best_weight = weight
            best_index = i

    return classes_list[best_index]


def _threshold_between(left, right):
    left = float(left)
    right = float(right)

    if math.isnan(left) or math.isnan(right):
        return left
    if left == -math.inf:
        return left
    if right == math.inf:
        return left

    threshold = left + (right - left) / 2.0
    if math.isnan(threshold) or threshold < left or threshold >= right:
        threshold = left
    return float(threshold)