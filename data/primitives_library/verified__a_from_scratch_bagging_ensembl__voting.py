import numpy as np

__all__ = ["majority_vote", "soft_vote"]


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


def majority_vote(pred_matrix, classes) -> np.ndarray:
    pred = _as_prediction_matrix(pred_matrix)
    classes_arr = _as_1d_classes(classes)
    ordered_classes = _ordered_labels(classes_arr)

    n_estimators, n_samples = pred.shape
    hashed, unhashable, nan_index = _build_class_lookup(ordered_classes)

    winner_indices = np.zeros(n_samples, dtype=np.intp)
    best_counts = np.zeros(n_samples, dtype=np.intp)
    counts_by_sample = [None] * n_samples

    for estimator_idx in range(n_estimators):
        row = pred[estimator_idx]
        for sample_idx in range(n_samples):
            class_index = _lookup_class_index(
                row[sample_idx], hashed, unhashable, nan_index
            )
            if class_index is None:
                continue

            sample_counts = counts_by_sample[sample_idx]
            if sample_counts is None:
                sample_counts = {}
                counts_by_sample[sample_idx] = sample_counts

            new_count = sample_counts.get(class_index, 0) + 1
            sample_counts[class_index] = new_count

            if (
                new_count > best_counts[sample_idx]
                or (
                    new_count == best_counts[sample_idx]
                    and class_index < winner_indices[sample_idx]
                )
            ):
                best_counts[sample_idx] = new_count
                winner_indices[sample_idx] = class_index

    winners = [ordered_classes[int(index)] for index in winner_indices]
    return _array_from_labels(winners, classes)


def soft_vote(proba_list, classes) -> np.ndarray:
    classes_arr = _as_1d_classes(classes)
    n_classes = classes_arr.size

    try:
        iterator = iter(proba_list)
    except TypeError as exc:
        raise ValueError("proba_list must be an iterable of probability arrays") from exc

    total = None
    expected_shape = None
    count = 0

    for proba in iterator:
        try:
            arr = np.asarray(proba, dtype=float)
        except Exception as exc:
            raise ValueError("each probability array must be numeric and array-like") from exc

        if arr.ndim != 2:
            raise ValueError("each probability array must have shape (n_samples, n_classes)")
        if arr.shape[1] != n_classes:
            raise ValueError("probability array class dimension must match len(classes)")

        if expected_shape is None:
            expected_shape = arr.shape
            total = np.array(arr, dtype=float, copy=True)
        else:
            if arr.shape != expected_shape:
                raise ValueError("all probability arrays must have the same shape")
            total += arr

        count += 1

    if count == 0:
        raise ValueError("proba_list must contain at least one probability array")

    mean_proba = total / float(count)
    indices = np.argmax(mean_proba, axis=1)
    winners = [classes_arr[int(i)] for i in indices]

    return _array_from_labels(winners, classes)