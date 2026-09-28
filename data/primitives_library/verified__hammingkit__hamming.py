"""Hamming distance utilities."""

from __future__ import annotations

__all__ = ["hamming", "hamming_bits"]


def hamming(a: str, b: str) -> int:
    """Return the Hamming distance between two equal-length strings.

    The Hamming distance is the number of positions at which corresponding
    characters differ.

    Raises:
        ValueError: If the two strings do not have the same length.
    """
    if len(a) != len(b):
        raise ValueError("hamming distance is defined only for equal-length strings")

    distance = 0
    for left, right in zip(a, b):
        if left != right:
            distance += 1
    return distance


def hamming_bits(x: int, y: int) -> int:
    """Return the number of differing bits between two integers.

    This is the population count of ``x ^ y``.
    """
    value = x ^ y

    if value < 0:
        value = -value

    count = 0
    while value:
        value &= value - 1
        count += 1
    return count