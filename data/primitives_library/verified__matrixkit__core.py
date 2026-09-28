"""Basic matrix operations for matrixkit.

This module implements small, dependency-free matrix primitives using nested
Python lists. It intentionally avoids NumPy-style operations so the behavior is
transparent and easy to inspect.
"""

__all__ = ["matmul", "transpose", "identity"]


def _shape(A):
    """Return (rows, cols) for a rectangular nested-list matrix.

    Raises:
        ValueError: If A is not a rectangular nested list.
    """
    if not isinstance(A, list):
        raise ValueError("matrix must be a list of rows")

    rows = len(A)
    if rows == 0:
        return 0, 0

    if not isinstance(A[0], list):
        raise ValueError("matrix must be a nested list")

    cols = len(A[0])
    for row in A:
        if not isinstance(row, list):
            raise ValueError("matrix must be a nested list")
        if len(row) != cols:
            raise ValueError("matrix rows must all have the same length")

    return rows, cols


def transpose(A):
    """Return the transpose of matrix A as a new nested list."""
    rows, cols = _shape(A)
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def matmul(A, B):
    """Multiply two matrices represented as nested lists.

    Args:
        A: Left matrix, shaped m x n.
        B: Right matrix, shaped n x p.

    Returns:
        The matrix product A @ B, shaped m x p.

    Raises:
        ValueError: If either matrix is malformed or shapes are incompatible.
    """
    a_rows, a_cols = _shape(A)
    b_rows, _ = _shape(B)

    if a_cols != b_rows:
        raise ValueError("matrix shapes are not aligned for multiplication")

    B_t = transpose(B)
    result = []

    for i in range(a_rows):
        result.append([])
        for j in range(len(B_t)):
            total = 0
            for k in range(a_cols):
                total += B_t[j][k] * A[i][k]
            result[i].append(total)

    return result


def identity(n) -> list:
    """Return the n x n identity matrix as a nested list."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")

    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]