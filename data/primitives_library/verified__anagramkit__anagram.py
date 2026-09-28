"""Anagram grouping utilities."""

from collections import defaultdict


def _normalized_key(value: str) -> tuple:
    """Return a canonical key for anagram comparison."""
    return tuple(sorted(ch for ch in value.casefold() if ch != " "))


def is_anagram(a: str, b: str) -> bool:
    """Return True if two strings are anagrams, case-insensitively ignoring spaces."""
    return _normalized_key(a) == _normalized_key(b)


def group_anagrams(words: list) -> list:
    """Group words into anagram groups.

    Each group is sorted, and the resulting groups are sorted by their first word.
    """
    groups = defaultdict(list)

    for word in words:
        groups[_normalized_key(word)].append(word)

    result = []
    for group in groups.values():
        result.append(sorted(group))

    return sorted(result, key=lambda group: group[0] if group else "")