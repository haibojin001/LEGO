__all__ = ["LRUCache"]


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        if not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")

        self.capacity = capacity
        self._items = {}

        self._root = _Node(None, None)
        self._root.prev = self._root
        self._root.next = self._root

    def get(self, key):
        node = self._items.get(key)
        if node is None:
            return None

        self._move_to_most_recent(node)
        return node.value

    def put(self, key, value):
        if self.capacity == 0:
            return

        node = self._items.get(key)
        if node is not None:
            node.value = value
            self._move_to_most_recent(node)
            return

        node = _Node(key, value)
        self._items[key] = node
        self._append_most_recent(node)

        if len(self._items) > self.capacity:
            self._evict_least_recent()

    def __len__(self):
        return len(self._items)

    def keys(self) -> list:
        result = []
        node = self._root.next

        while node is not self._root:
            result.append(node.key)
            node = node.next

        return result

    def _remove_node(self, node):
        prev_node = node.prev
        next_node = node.next

        prev_node.next = next_node
        next_node.prev = prev_node

        node.prev = None
        node.next = None

    def _append_most_recent(self, node):
        last = self._root.prev

        node.prev = last
        node.next = self._root
        last.next = node
        self._root.prev = node

    def _move_to_most_recent(self, node):
        if node.next is self._root:
            return

        self._remove_node(node)
        self._append_most_recent(node)

    def _evict_least_recent(self):
        node = self._root.next
        if node is self._root:
            return

        self._remove_node(node)
        del self._items[node.key]