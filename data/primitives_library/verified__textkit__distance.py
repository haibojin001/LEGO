"""String edit distances for :mod:`textkit`.

This module provides dynamic-programming implementations of common string
distance/subsequence measures using only the Python standard library.
"""

__all__ = ["levenshtein", "lcs_len"]


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings.

    The distance is the minimum number of single-character insertions,
    deletions, and substitutions required to transform ``a`` into ``b``.
    """
    if a == b:
        return 0

    len_a = len(a)
    len_b = len(b)

    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    if len_b > len_a:
        a, b = b, a
        len_a, len_b = len_b, len_a

    previous = list(range(len_b + 1))

    for i in range(1, len_a + 1):
        current = [i] + [0] * len_b
        char_a = a[i - 1]

        for j in range(1, len_b + 1):
            if char_a == b[j - 1]:
                current[j] = previous[j - 1]
            else:
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1
                substitution = previous[j - 1] + 1
                current[j] = min(insertion, deletion, substitution)

        previous = current

    return previous[len_b]


def lcs_len(a: str, b: str) -> int:
    """Return the length of the longest common subsequence of two strings.

    A subsequence preserves relative order but does not require characters to be
    contiguous.
    """
    if not a or not b:
        return 0

    if len(b) > len(a):
        a, b = b, a

    previous = [0] * (len(b) + 1)

    for char_a in a:
        current = [0] * (len(b) + 1)

        for j in range(1, len(b) + 1):
            if char_a == b[j - 1]:
                current[j] = previous[j - 1] + 1
            elif previous[j] >= current[j - 1]:
                current[j] = previous[j]
            else:
                current[j] = current[j - 1]

        previous = current

    return previous[len(b)]