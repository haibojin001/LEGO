"""Interval set operations for half-open intervals [start, end)."""

__all__ = ["merge", "intersect", "subtract", "total_length"]


def _normalized(intervals):
    """Return sorted, merged intervals as mutable [start, end] pairs."""
    items = []

    for interval in intervals:
        try:
            start, end = interval
        except Exception as exc:
            raise ValueError("each interval must contain exactly two values") from exc

        if end < start:
            raise ValueError("interval end must be greater than or equal to start")

        if end == start:
            continue

        items.append([start, end])

    if not items:
        return []

    items.sort(key=lambda pair: (pair[0], pair[1]))

    merged = [items[0]]
    for start, end in items[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])

    return merged


def merge(intervals) -> list:
    """Merge overlapping or adjacent half-open intervals, sorted by start."""
    return _normalized(intervals)


def intersect(a, b) -> list:
    """Return the intersection of two interval sets."""
    left = _normalized(a)
    right = _normalized(b)

    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        a_start, a_end = left[i]
        b_start, b_end = right[j]

        start = a_start if a_start >= b_start else b_start
        end = a_end if a_end <= b_end else b_end

        if start < end:
            result.append([start, end])

        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result


def subtract(a, b) -> list:
    """Return the interval set difference a - b."""
    left = _normalized(a)
    right = _normalized(b)

    if not left or not right:
        return left

    result = []
    j = 0

    for start, end in left:
        current = start

        while j < len(right) and right[j][1] <= current:
            j += 1

        k = j
        while k < len(right) and right[k][0] < end:
            cut_start, cut_end = right[k]

            if cut_start > current:
                result.append([current, cut_start])

            if cut_end > current:
                current = cut_end

            if current >= end:
                break

            k += 1

        if current < end:
            result.append([current, end])

    return result


def total_length(intervals) -> float:
    """Return the total length of the union of the intervals."""
    return float(sum(end - start for start, end in _normalized(intervals)))