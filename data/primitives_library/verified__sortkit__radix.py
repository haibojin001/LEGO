"""Non-comparison integer sorting algorithms for sortkit."""

__all__ = ["radix_sort"]


def radix_sort(xs) -> list:
    """Return the non-negative integers from *xs* sorted in ascending order.

    Uses a stable least-significant-digit radix sort with base 256.
    """
    arr = list(xs)
    n = len(arr)

    if n < 2:
        for value in arr:
            if not isinstance(value, int):
                raise TypeError("radix_sort expects non-negative integers")
            if value < 0:
                raise ValueError("radix_sort expects non-negative integers")
        return arr

    max_value = 0
    for value in arr:
        if not isinstance(value, int):
            raise TypeError("radix_sort expects non-negative integers")
        if value < 0:
            raise ValueError("radix_sort expects non-negative integers")
        if value > max_value:
            max_value = value

    shift = 0
    while max_value >> shift:
        counts = [0] * 256

        for value in arr:
            counts[(value >> shift) & 0xFF] += 1

        total = 0
        for i in range(256):
            count = counts[i]
            counts[i] = total
            total += count

        out = [0] * n
        for value in arr:
            byte = (value >> shift) & 0xFF
            out[counts[byte]] = value
            counts[byte] += 1

        arr = out
        shift += 8

    return arr