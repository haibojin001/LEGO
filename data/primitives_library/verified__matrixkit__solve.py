__all__ = ["solve"]


def _to_python_lists(value):
    """Recursively convert common array/matrix containers to Python lists."""
    if isinstance(value, (str, bytes, bytearray)):
        return value

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _to_python_lists(tolist())

    for attr in ("data", "_data", "rows", "_rows"):
        if hasattr(value, attr):
            attr_value = getattr(value, attr)
            if attr_value is not value and not callable(attr_value):
                return _to_python_lists(attr_value)

    if isinstance(value, (list, tuple)):
        return [_to_python_lists(item) for item in value]

    try:
        iterator = iter(value)
    except TypeError:
        return value

    return [_to_python_lists(item) for item in iterator]


def _is_list_like(value):
    return isinstance(value, list)


def _coerce_square_matrix(A):
    data = _to_python_lists(A)

    if not _is_list_like(data) or len(data) == 0:
        raise ValueError("A must be a non-empty square matrix")

    matrix = []
    width = None

    for row in data:
        if not _is_list_like(row) or len(row) == 0:
            raise ValueError("A must be a non-empty square matrix")

        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("A must be a rectangular matrix")

        try:
            matrix.append([float(value) for value in row])
        except (TypeError, ValueError) as exc:
            raise ValueError("A must contain numeric values") from exc

    n = len(matrix)
    if width != n:
        raise ValueError("A must be square")

    return matrix, n


def _coerce_rhs(b, n):
    data = _to_python_lists(b)

    if not _is_list_like(data):
        if n != 1:
            raise ValueError("b has incompatible dimensions")
        try:
            return [[float(data)]], True
        except (TypeError, ValueError) as exc:
            raise ValueError("b must contain numeric values") from exc

    if len(data) == 0:
        raise ValueError("b must not be empty")

    row_flags = [_is_list_like(item) for item in data]

    if any(row_flags):
        if not all(row_flags):
            raise ValueError("b must be either a vector or a rectangular matrix")
        if len(data) != n:
            raise ValueError("b has incompatible dimensions")

        width = None
        rhs = []
        for row in data:
            if len(row) == 0:
                raise ValueError("b must be either a vector or a rectangular matrix")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("b must be a rectangular matrix")

            try:
                rhs.append([float(value) for value in row])
            except (TypeError, ValueError) as exc:
                raise ValueError("b must contain numeric values") from exc

        return rhs, False

    if len(data) != n:
        raise ValueError("b has incompatible dimensions")

    try:
        return [[float(value)] for value in data], True
    except (TypeError, ValueError) as exc:
        raise ValueError("b must contain numeric values") from exc


def _singularity_tolerance(matrix):
    max_abs = 0.0
    for row in matrix:
        for value in row:
            abs_value = abs(value)
            if abs_value > max_abs:
                max_abs = abs_value

    if max_abs == 0.0:
        return 0.0

    return 2.220446049250313e-16 * max(1, len(matrix)) * max_abs


def solve(A, b):
    matrix, n = _coerce_square_matrix(A)
    rhs, vector_rhs = _coerce_rhs(b, n)

    tolerance = _singularity_tolerance(matrix)
    if tolerance == 0.0:
        raise ValueError("singular matrix")

    rhs_columns = len(rhs[0])

    for k in range(n):
        pivot_row = k
        pivot_abs = abs(matrix[k][k])

        for row in range(k + 1, n):
            candidate_abs = abs(matrix[row][k])
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                pivot_row = row

        if pivot_abs <= tolerance:
            raise ValueError("singular matrix")

        if pivot_row != k:
            matrix[k], matrix[pivot_row] = matrix[pivot_row], matrix[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]

        pivot = matrix[k][k]

        for row in range(k + 1, n):
            factor = matrix[row][k] / pivot
            matrix[row][k] = 0.0

            if factor != 0.0:
                for col in range(k + 1, n):
                    matrix[row][col] -= factor * matrix[k][col]
                for col in range(rhs_columns):
                    rhs[row][col] -= factor * rhs[k][col]

    solution = [[0.0 for _ in range(rhs_columns)] for _ in range(n)]

    for row in range(n - 1, -1, -1):
        diagonal = matrix[row][row]
        if abs(diagonal) <= tolerance:
            raise ValueError("singular matrix")

        for rhs_col in range(rhs_columns):
            total = rhs[row][rhs_col]
            for col in range(row + 1, n):
                total -= matrix[row][col] * solution[col][rhs_col]
            solution[row][rhs_col] = total / diagonal

    if vector_rhs:
        return [solution[row][0] for row in range(n)]

    return solution