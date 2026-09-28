"""Longest common subsequence utilities."""

__all__ = ["lcs_len", "lcs"]


def lcs_len(a: str, b: str) -> int:
    """Return the length of the longest common subsequence of *a* and *b*."""
    if not a or not b:
        return 0

    if len(b) > len(a):
        a, b = b, a

    previous = [0] * (len(b) + 1)

    for ca in a:
        current = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = previous[j] if previous[j] >= current[j - 1] else current[j - 1]
        previous = current

    return previous[-1]


def lcs(a: str, b: str) -> str:
    """Return one longest common subsequence of *a* and *b*.

    Ties during reconstruction are resolved by moving left in the dynamic
    programming table, matching the standard deterministic backtracking used
    by the adapted primitive.
    """
    m = len(a)
    n = len(b)

    if m == 0 or n == 0:
        return ""

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        ca = a[i - 1]
        row = dp[i]
        previous_row = dp[i - 1]
        for j in range(1, n + 1):
            if ca == b[j - 1]:
                row[j] = previous_row[j - 1] + 1
            else:
                up = previous_row[j]
                left = row[j - 1]
                row[j] = up if up >= left else left

    i = m
    j = n
    result = []

    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    result.reverse()
    return "".join(result)