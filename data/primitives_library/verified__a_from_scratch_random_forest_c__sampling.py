"""Bootstrap resampling utilities."""

import numpy as np

__all__ = ["bootstrap_sample"]


def bootstrap_sample(n, seed=0) -> np.ndarray:
    """Return n bootstrap indices drawn with replacement from range(n)."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be a non-negative integer")

    n = int(n)
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n, dtype=np.int64)