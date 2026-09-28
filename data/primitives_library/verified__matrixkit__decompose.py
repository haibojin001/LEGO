"""LU decomposition and determinant routines for matrixkit."""

from matrixkit.core import _shape, identity, matmul

__all__ = ["lu", "det"]


class _Matrix(list):
    """Small list subclass that also supports the @ operator."""

    def __matmul__(self, other):
        return _wrap_matrix(_matmul_plain(self, other))

    def __rmatmul__(self, other):
        return _wrap_matrix(_matmul_plain(other, self))


def _wrap_matrix(matrix):
    return _Matrix([list(row) for row in matrix])


def _to_nested_lists(A, *, coerce_float=False):
    if hasattr(A, "tolist"):
        A = A.tolist()

    if isinstance(A, (str, bytes)):
        raise TypeError("matrix must be a two-dimensional numeric sequence")

    try:
        raw_rows = list(A)
    except TypeError as exc:
        raise TypeError("matrix must be a two-dimensional numeric sequence") from exc

    rows = []
    expected_cols = None

    for raw_row in raw_rows:
        if hasattr(raw_row, "tolist"):
            raw_row = raw_row.tolist()

        if isinstance(raw_row, (str, bytes)):
            raise TypeError("matrix rows must be numeric sequences")

        try:
            row = list(raw_row)
        except TypeError as exc:
            raise ValueError("matrix must be two-dimensional") from exc

        if expected_cols is None:
            expected_cols = len(row)
        elif len(row) != expected_cols:
            raise ValueError("matrix rows must all have the same length")

        if coerce_float:
            try:
                row = [float(x) for x in row]
            except (TypeError, ValueError) as exc:
                raise TypeError("matrix entries must be numeric") from exc

        rows.append(row)

    return rows


def _dimensions(matrix):
    try:
        shp = _shape(matrix)
        if len(shp) == 2:
            return int(shp[0]), int(shp[1])
    except Exception:
        pass

    return len(matrix), (len(matrix[0]) if matrix else 0)


def _as_square_float_matrix(A):
    matrix = _to_nested_lists(A, coerce_float=True)
    rows, cols = _dimensions(matrix)

    if rows != cols:
        raise ValueError("matrix must be square")

    return matrix


def _identity_matrix(n):
    try:
        I = _to_nested_lists(identity(n), coerce_float=True)
        rows, cols = _dimensions(I)
        if rows == n and cols == n:
            return _wrap_matrix(I)
    except Exception:
        pass

    return _wrap_matrix(
        [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    )


def _matmul_plain(A, B):
    A = _to_nested_lists(A, coerce_float=False)
    B = _to_nested_lists(B, coerce_float=False)

    try:
        return matmul(A, B)
    except Exception:
        pass

    a_rows, a_cols = _dimensions(A)
    b_rows, b_cols = _dimensions(B)

    if a_cols != b_rows:
        raise ValueError("incompatible matrix dimensions for multiplication")

    if a_rows == 0:
        return []

    result = []
    for i in range(a_rows):
        out_row = []
        for j in range(b_cols):
            total = 0
            for k in range(a_cols):
                total += A[i][k] * B[k][j]
            out_row.append(total)
        result.append(out_row)

    return result


def _lu_decompose(A):
    U = _as_square_float_matrix(A)
    n = len(U)

    P = _identity_matrix(n)
    L = _identity_matrix(n)
    U = _wrap_matrix(U)

    pivot_sign = 1.0

    for k in range(n):
        pivot_row = k
        pivot_abs = abs(U[k][k])

        for i in range(k + 1, n):
            candidate_abs = abs(U[i][k])
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                pivot_row = i

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]

            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]

            pivot_sign = -pivot_sign

        if abs(U[k][k]) == 0:
            continue

        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            U[i][k] = 0.0

            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]

    return _wrap_matrix(P), _wrap_matrix(L), _wrap_matrix(U), pivot_sign


def lu(A):
    """Return (P, L, U) from LU decomposition with partial pivoting.

    The returned matrices satisfy P @ A == L @ U, up to floating point
    roundoff.
    """
    P, L, U, _ = _lu_decompose(A)
    return P, L, U


def det(A) -> float:
    """Return the determinant of a square matrix using LU decomposition."""
    _, _, U, pivot_sign = _lu_decompose(A)

    product = 1.0
    for i in range(len(U)):
        product *= U[i][i]

    value = pivot_sign * product
    if value == 0:
        return 0.0
    return float(value)