"""
Suffix array construction, LCP computation, and suffix-array based search.

Public API:
    suffix_array(s)
    lcp_array(s, sa)
    count_occurrences(s, sa, pattern)
"""

__all__ = ["suffix_array", "lcp_array", "count_occurrences"]


def suffix_array(s: str) -> list:
    """Return the indices of all suffixes of *s* in lexicographic order."""
    n = len(s)
    if n <= 1:
        return list(range(n))

    sa = list(range(n))
    rank = [ord(ch) for ch in s]
    tmp = [0] * n
    k = 1

    while k < n:
        old_rank = rank

        def key(index):
            return (
                old_rank[index],
                old_rank[index + k] if index + k < n else -1,
            )

        sa.sort(key=key)

        tmp[sa[0]] = 0
        classes = 0

        for i in range(1, n):
            prev = sa[i - 1]
            curr = sa[i]

            prev_key = (
                old_rank[prev],
                old_rank[prev + k] if prev + k < n else -1,
            )
            curr_key = (
                old_rank[curr],
                old_rank[curr + k] if curr + k < n else -1,
            )

            if curr_key != prev_key:
                classes += 1
            tmp[curr] = classes

        rank, tmp = tmp, rank

        if classes == n - 1:
            break

        k <<= 1

    return sa


def lcp_array(s: str, sa: list) -> list:
    """
    Return the Kasai LCP array for *s* and its suffix array *sa*.

    lcp[i] is the length of the longest common prefix of the suffixes starting
    at sa[i - 1] and sa[i]. By convention, lcp[0] is 0.
    """
    n = len(s)
    if len(sa) != n:
        raise ValueError("suffix array length must match string length")

    if n == 0:
        return []

    rank = [-1] * n
    for pos, index in enumerate(sa):
        if not isinstance(index, int):
            raise ValueError("suffix array indices must be integers")
        if index < 0 or index >= n:
            raise ValueError("suffix array index out of range")
        if rank[index] != -1:
            raise ValueError("suffix array contains duplicate indices")
        rank[index] = pos

    lcp = [0] * n
    h = 0

    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue

        j = sa[r - 1]

        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1

        lcp[r] = h

        if h:
            h -= 1

    return lcp


def count_occurrences(s: str, sa: list, pattern: str) -> int:
    """
    Count overlapping occurrences of *pattern* in *s* using binary search over
    the suffix array *sa*.
    """
    n = len(s)
    if len(sa) != n:
        raise ValueError("suffix array length must match string length")

    if pattern == "":
        return n + 1

    left = _lower_bound_prefix(s, sa, pattern)
    right = _upper_bound_prefix(s, sa, pattern, left)
    return right - left


def _lower_bound_prefix(s, sa, pattern):
    lo = 0
    hi = len(sa)

    while lo < hi:
        mid = (lo + hi) // 2
        if _compare_suffix_prefix(s, sa[mid], pattern) < 0:
            lo = mid + 1
        else:
            hi = mid

    return lo


def _upper_bound_prefix(s, sa, pattern, start):
    lo = start
    hi = len(sa)

    while lo < hi:
        mid = (lo + hi) // 2
        if _compare_suffix_prefix(s, sa[mid], pattern) <= 0:
            lo = mid + 1
        else:
            hi = mid

    return lo


def _compare_suffix_prefix(s, start, pattern):
    """
    Compare s[start:start + len(pattern)] with pattern.

    Return:
        -1 if the suffix prefix is lexicographically smaller than pattern,
         0 if pattern is a prefix of s[start:],
         1 if the suffix prefix is lexicographically greater than pattern.
    """
    n = len(s)
    m = len(pattern)

    if not isinstance(start, int) or start < 0 or start >= n:
        raise ValueError("suffix array index out of range")

    i = 0
    while i < m and start + i < n:
        a = s[start + i]
        b = pattern[i]

        if a < b:
            return -1
        if a > b:
            return 1

        i += 1

    if i == m:
        return 0

    return -1