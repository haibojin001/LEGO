"""Logistic loss utilities for gbkit."""

import numpy as np

__all__ = ["sigmoid", "neg_gradient", "log_loss"]


def sigmoid(z) -> np.ndarray:
    """Compute the logistic sigmoid stably for scalars or arrays.

    Uses separate formulas for nonnegative and negative inputs so that large
    magnitudes do not overflow ``exp``.
    """
    arr = np.asarray(z, dtype=float)
    out = np.empty(arr.shape, dtype=float)

    nonnegative = arr >= 0

    if np.any(nonnegative):
        x = arr[nonnegative]
        out[nonnegative] = 1.0 / (1.0 + np.exp(-x))

    negative = ~nonnegative
    if np.any(negative):
        x = arr[negative]
        exp_x = np.exp(x)
        out[negative] = exp_x / (1.0 + exp_x)

    return out


def neg_gradient(y, raw) -> np.ndarray:
    """Return pseudo-residuals for binary logistic loss: y - sigmoid(raw)."""
    return np.asarray(y, dtype=float) - sigmoid(raw)


def log_loss(y, raw) -> float:
    """Return mean binary logistic loss from raw scores.

    For labels ``y`` in {0, 1} and raw scores ``raw`` (logits), this computes

        mean(log(1 + exp(raw)) - y * raw)

    using a stable piecewise form.
    """
    y_arr = np.asarray(y, dtype=float)
    raw_arr = np.asarray(raw, dtype=float)
    y_arr, raw_arr = np.broadcast_arrays(y_arr, raw_arr)

    losses = np.empty(raw_arr.shape, dtype=float)

    finite = np.isfinite(raw_arr)
    nonnegative_finite = finite & (raw_arr >= 0)
    negative_finite = finite & (raw_arr < 0)

    if np.any(nonnegative_finite):
        r = raw_arr[nonnegative_finite]
        t = y_arr[nonnegative_finite]
        losses[nonnegative_finite] = (1.0 - t) * r + np.log1p(np.exp(-r))

    if np.any(negative_finite):
        r = raw_arr[negative_finite]
        t = y_arr[negative_finite]
        losses[negative_finite] = np.log1p(np.exp(r)) - t * r

    pos_inf = np.isposinf(raw_arr)
    if np.any(pos_inf):
        coef = 1.0 - y_arr[pos_inf]
        vals = np.empty(coef.shape, dtype=float)
        vals[coef == 0.0] = 0.0
        vals[coef > 0.0] = np.inf
        vals[coef < 0.0] = -np.inf
        vals[np.isnan(coef)] = np.nan
        losses[pos_inf] = vals

    neg_inf = np.isneginf(raw_arr)
    if np.any(neg_inf):
        coef = y_arr[neg_inf]
        vals = np.empty(coef.shape, dtype=float)
        vals[coef == 0.0] = 0.0
        vals[coef > 0.0] = np.inf
        vals[coef < 0.0] = -np.inf
        vals[np.isnan(coef)] = np.nan
        losses[neg_inf] = vals

    nan_raw = np.isnan(raw_arr)
    if np.any(nan_raw):
        losses[nan_raw] = np.nan

    return float(np.mean(losses))