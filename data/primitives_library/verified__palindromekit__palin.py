"""Palindrome utilities for palindromekit."""

__all__ = ["is_palindrome", "longest_palindrome"]


def is_palindrome(s: str, ignore_case: bool = True, alnum_only: bool = True) -> bool:
    """Return True if *s* is a palindrome under the requested normalization."""
    if alnum_only:
        s = "".join(ch for ch in s if ch.isalnum())

    if ignore_case:
        s = s.casefold()

    return s == s[::-1]


def longest_palindrome(s: str) -> str:
    """Return the longest contiguous palindromic substring of *s*.

    If multiple substrings have the same maximum length, the leftmost one is
    returned.
    """
    if not s:
        return ""

    n = len(s)
    left = 0
    right = 0

    def extend(i: int, j: int) -> tuple[int, int]:
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
        return i + 1, j - 1

    for i in range(n):
        l1, r1 = extend(i, i)
        if r1 - l1 > right - left:
            left, right = l1, r1

        if i + 1 < n and s[i] == s[i + 1]:
            l2, r2 = extend(i, i + 1)
            if r2 - l2 > right - left:
                left, right = l2, r2

    return s[left:right + 1]