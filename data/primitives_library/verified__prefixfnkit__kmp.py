"""Prefix function, Z-function, and KMP string matching utilities."""

__all__ = ["prefix_function", "z_function", "find_all"]


def prefix_function(s: str) -> list:
    """Return the prefix-function (pi array) for *s*.

    pi[i] is the length of the longest proper prefix of s[:i + 1] that is also
    a suffix of s[:i + 1].
    """
    n = len(s)
    pi = [0] * n

    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j

    return pi


def z_function(s: str) -> list:
    """Return the Z-function array for *s*.

    z[i] is the length of the longest substring starting at i that is also a
    prefix of s. By convention, z[0] is 0.
    """
    n = len(s)
    z = [0] * n
    left = 0
    right = 0

    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1

    return z


def find_all(text: str, pattern: str) -> list:
    """Return start indices of every possibly overlapping match of pattern in text.

    Uses the Knuth-Morris-Pratt algorithm for non-empty patterns. An empty
    pattern is considered to match at every position, including len(text).
    """
    if pattern == "":
        return list(range(len(text) + 1))

    pi = prefix_function(pattern)
    matches = []
    j = 0

    for i, ch in enumerate(text):
        while j > 0 and ch != pattern[j]:
            j = pi[j - 1]

        if ch == pattern[j]:
            j += 1

        if j == len(pattern):
            matches.append(i - len(pattern) + 1)
            j = pi[j - 1]

    return matches