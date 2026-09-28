"""Bootstrap sampling and out-of-bag index utilities."""

import numpy as np

__all__ = ["bootstrap_sample", "oob_indices"]


def _validate_n(n) -> int:
    """Validate and normalize a sample count."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be a non-negative integer")

    n = int(n)
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    return n


def bootstrap_sample(n, seed=0) -> np.ndarray:
    """Return n bootstrap indices drawn with replacement from range(n).

    The result is deterministic for a fixed ``n`` and ``seed``.
    """
    n = _validate_n(n)

    if n == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n, dtype=np.int64)


def oob_indices(n, in_bag) -> np.ndarray:
    """Return indices in range(n) that are not present in ``in_bag``."""
    n = _validate_n(n)

    arr = np.asarray(in_bag)
    if arr.size == 0:
        return np.arange(n, dtype=np.int64)

    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("in_bag must be an array of integer indices")

    flat = arr.ravel()

    if np.any(flat < 0) or np.any(flat >= n):
        raise ValueError("in_bag contains indices outside range(n)")

    present = np.zeros(n, dtype=bool)
    present[flat.astype(np.intp, copy=False)] = True
    return np.nonzero(~present)[0].astype(np.int64, copy=False)