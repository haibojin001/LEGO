"""SAMME sample-weight utilities."""

import math

import numpy as np


def _as_1d_checked(y_true, y_pred, sample_weight):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    sw = np.asarray(sample_weight, dtype=float).ravel()

    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of samples")
    if sw.shape[0] != yt.shape[0]:
        raise ValueError("sample_weight must contain one weight per sample")
    if sw.shape[0] == 0:
        raise ValueError("at least one sample is required")
    if not np.all(np.isfinite(sw)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(sw < 0):
        raise ValueError("sample_weight must be non-negative")

    total = float(np.sum(sw))
    if total <= 0.0:
        raise ValueError("sample_weight must sum to a positive value")

    return yt, yp, sw, total


def weighted_error(y_true, y_pred, sample_weight) -> float:
    """Return the weighted misclassification rate."""
    yt, yp, sw, total = _as_1d_checked(y_true, y_pred, sample_weight)
    wrong = yt != yp
    return float(np.sum(sw[wrong]) / total)


def estimator_weight(err, n_classes) -> float:
    """Return the SAMME estimator weight."""
    if not isinstance(n_classes, (int, np.integer)):
        raise TypeError("n_classes must be an integer")
    if n_classes <= 1:
        raise ValueError("n_classes must be greater than 1")

    err = float(err)
    if math.isnan(err):
        raise ValueError("err must not be NaN")
    if err < 0.0 or err > 1.0:
        raise ValueError("err must be in the interval [0, 1]")

    class_term = math.log(float(n_classes - 1))

    if err == 0.0:
        return math.inf
    if err == 1.0:
        return -math.inf

    return float(math.log((1.0 - err) / err) + class_term)


def update_sample_weights(sample_weight, y_true, y_pred, alpha) -> np.ndarray:
    """Multiply wrong-sample weights by exp(alpha), then renormalize to sum 1."""
    yt, yp, sw, _ = _as_1d_checked(y_true, y_pred, sample_weight)

    alpha = float(alpha)
    if math.isnan(alpha):
        raise ValueError("alpha must not be NaN")

    wrong = yt != yp

    if alpha == math.inf:
        new_sw = np.zeros_like(sw, dtype=float)
        wrong_sum = float(np.sum(sw[wrong]))
        if wrong_sum <= 0.0:
            new_sw = sw.astype(float, copy=True)
            new_sw /= float(np.sum(new_sw))
            return new_sw
        new_sw[wrong] = sw[wrong] / wrong_sum
        return new_sw

    if alpha == -math.inf:
        new_sw = sw.astype(float, copy=True)
        new_sw[wrong] = 0.0
        total = float(np.sum(new_sw))
        if total <= 0.0:
            raise ValueError("renormalized sample weights sum to zero")
        new_sw /= total
        return new_sw

    scale = max(0.0, alpha)
    correct_factor = math.exp(-scale)
    wrong_factor = math.exp(alpha - scale)

    new_sw = sw.astype(float, copy=True)
    new_sw[wrong] *= wrong_factor
    new_sw[~wrong] *= correct_factor

    total = float(np.sum(new_sw))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("renormalized sample weights must have a positive finite sum")

    new_sw /= total
    return new_sw