class _BinaryHeap:
    """Internal binary heap implementation parameterized by priority order."""

    __slots__ = ("_data",)

    def __init__(self, items=None):
        if items is None:
            self._data = []
        else:
            self._data = list(items)
            self._heapify()

    def _comes_before(self, a, b):
        raise NotImplementedError

    def _heapify(self):
        for index in range((len(self._data) // 2) - 1, -1, -1):
            self._sift_down(index)

    def _sift_up(self, index):
        data = self._data
        item = data[index]

        while index > 0:
            parent = (index - 1) // 2
            parent_item = data[parent]

            if not self._comes_before(item, parent_item):
                break

            data[index] = parent_item
            index = parent

        data[index] = item

    def _sift_down(self, index):
        data = self._data
        size = len(data)
        item = data[index]
        half = size // 2

        while index < half:
            left = (index * 2) + 1
            right = left + 1
            child = left
            child_item = data[left]

            if right < size and self._comes_before(data[right], child_item):
                child = right
                child_item = data[right]

            if not self._comes_before(child_item, item):
                break

            data[index] = child_item
            index = child

        data[index] = item

    def push(self, x):
        self._data.append(x)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        data = self._data

        if not data:
            raise IndexError("pop from empty heap")

        result = data[0]
        last = data.pop()

        if data:
            data[0] = last
            self._sift_down(0)

        return result

    def peek(self):
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def __len__(self):
        return len(self._data)


class MinHeap(_BinaryHeap):
    def __init__(self, items=None):
        super().__init__(items)

    def _comes_before(self, a, b):
        return a < b


class MaxHeap(_BinaryHeap):
    def __init__(self, items=None):
        super().__init__(items)

    def _comes_before(self, a, b):
        return a > b


__all__ = ("MinHeap", "MaxHeap")