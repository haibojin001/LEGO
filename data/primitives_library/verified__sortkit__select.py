"""Order-statistic selection algorithms."""

__all__ = ["quickselect"]


def quickselect(xs, k: int):
    """Return the 0-indexed kth smallest element of *xs*.

    The input is copied before selection, so the original iterable is not
    modified.  Raises ValueError if *k* is outside ``[0, len(xs))``.
    """
    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    arr = list(xs)
    n = len(arr)

    if k < 0 or k >= n:
        raise ValueError("k is out of range")

    left = 0
    right = n - 1

    while left <= right:
        pivot_index = _partition(arr, left, right)

        if pivot_index == k:
            return arr[pivot_index]
        if pivot_index > k:
            right = pivot_index - 1
        else:
            left = pivot_index + 1

    raise RuntimeError("quickselect failed to converge")


def _partition(arr, left, right):
    """Partition arr[left:right + 1] and return the pivot's final index."""
    pivot_slot = _median_of_three(arr, left, right)
    arr[pivot_slot], arr[right] = arr[right], arr[pivot_slot]

    pivot = arr[right]
    store = left

    for i in range(left, right):
        if arr[i] < pivot:
            arr[store], arr[i] = arr[i], arr[store]
            store += 1

    arr[store], arr[right] = arr[right], arr[store]
    return store


def _median_of_three(arr, left, right):
    """Return the index of the median among left, middle, and right."""
    middle = (left + right) // 2

    if arr[middle] < arr[left]:
        arr[left], arr[middle] = arr[middle], arr[left]
    if arr[right] < arr[left]:
        arr[left], arr[right] = arr[right], arr[left]
    if arr[right] < arr[middle]:
        arr[middle], arr[right] = arr[right], arr[middle]

    return middle