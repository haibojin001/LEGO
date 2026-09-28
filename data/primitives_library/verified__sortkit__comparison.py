"""Comparison-based sorting algorithms for sortkit."""

__all__ = ["merge_sort", "quick_sort", "heap_sort"]


def merge_sort(xs) -> list:
    """Return a new list containing the items from xs in stable ascending order."""
    src = list(xs)
    n = len(src)

    if n < 2:
        return src

    dest = [None] * n
    width = 1

    while width < n:
        step = width * 2

        for lo in range(0, n, step):
            mid = min(lo + width, n)
            hi = min(lo + step, n)

            i = lo
            j = mid
            k = lo

            while i < mid and j < hi:
                if src[j] < src[i]:
                    dest[k] = src[j]
                    j += 1
                else:
                    dest[k] = src[i]
                    i += 1
                k += 1

            while i < mid:
                dest[k] = src[i]
                i += 1
                k += 1

            while j < hi:
                dest[k] = src[j]
                j += 1
                k += 1

        src, dest = dest, src
        width = step

    return src


def quick_sort(xs) -> list:
    """Return a new list containing the items from xs in ascending order."""
    a = list(xs)
    n = len(a)

    if n < 2:
        return a

    stack = [(0, n - 1)]

    while stack:
        lo, hi = stack.pop()

        while lo < hi:
            if hi - lo <= 16:
                _insertion_sort_range(a, lo, hi)
                break

            pivot = a[_median_of_three_index(a, lo, (lo + hi) // 2, hi)]

            lt = lo
            i = lo
            gt = hi

            while i <= gt:
                if a[i] < pivot:
                    a[lt], a[i] = a[i], a[lt]
                    lt += 1
                    i += 1
                elif pivot < a[i]:
                    a[i], a[gt] = a[gt], a[i]
                    gt -= 1
                else:
                    i += 1

            left_lo = lo
            left_hi = lt - 1
            right_lo = gt + 1
            right_hi = hi

            left_size = left_hi - left_lo + 1
            right_size = right_hi - right_lo + 1

            if left_size < right_size:
                if right_lo < right_hi:
                    stack.append((right_lo, right_hi))
                lo, hi = left_lo, left_hi
            else:
                if left_lo < left_hi:
                    stack.append((left_lo, left_hi))
                lo, hi = right_lo, right_hi

    return a


def heap_sort(xs) -> list:
    """Return a new list containing the items from xs in ascending order."""
    heap = list(xs)
    n = len(heap)

    if n < 2:
        return heap

    for start in range((n // 2) - 1, -1, -1):
        _sift_down_min_heap(heap, start, n)

    result = []

    end = n
    while end > 0:
        result.append(heap[0])
        end -= 1

        if end:
            heap[0] = heap[end]
            _sift_down_min_heap(heap, 0, end)

    return result


def _insertion_sort_range(a, lo, hi):
    for i in range(lo + 1, hi + 1):
        value = a[i]
        j = i - 1

        while j >= lo and value < a[j]:
            a[j + 1] = a[j]
            j -= 1

        a[j + 1] = value


def _median_of_three_index(a, i, j, k):
    x = a[i]
    y = a[j]
    z = a[k]

    if x < y:
        if y < z:
            return j
        if x < z:
            return k
        return i

    if x < z:
        return i
    if y < z:
        return k
    return j


def _sift_down_min_heap(heap, start, end):
    root = start

    while True:
        child = (root * 2) + 1

        if child >= end:
            return

        right = child + 1

        if right < end and heap[right] < heap[child]:
            child = right

        if heap[child] < heap[root]:
            heap[root], heap[child] = heap[child], heap[root]
            root = child
        else:
            return