"""Edit-distance dynamic programming for levenshteinkit2."""

__all__ = ["levenshtein", "edit_matrix"]


def edit_matrix(a: str, b: str) -> list:
    """Return the full Levenshtein dynamic-programming table.

    The returned matrix has ``len(a) + 1`` rows and ``len(b) + 1`` columns.
    Cell ``matrix[i][j]`` is the minimum number of insertions, deletions, and
    substitutions needed to transform ``a[:i]`` into ``b[:j]``.
    """
    rows = len(a) + 1
    cols = len(b) + 1

    matrix = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        matrix[i][0] = i

    for j in range(cols):
        matrix[0][j] = j

    for i in range(1, rows):
        char_a = a[i - 1]
        previous_row = matrix[i - 1]
        current_row = matrix[i]

        for j in range(1, cols):
            if char_a == b[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                insertion = current_row[j - 1]
                deletion = previous_row[j]
                substitution = previous_row[j - 1]
                current_row[j] = min(insertion, deletion, substitution) + 1

    return matrix


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    return edit_matrix(a, b)[len(a)][len(b)]