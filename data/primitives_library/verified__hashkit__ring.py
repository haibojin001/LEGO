from bisect import bisect_left, insort
from hashkit.fnv import fnv1a_64

__all__ = ["HashRing"]


class HashRing:
    def __init__(self, nodes=None, vnodes: int = 100):
        if not isinstance(vnodes, int):
            raise TypeError("vnodes must be an int")
        if vnodes <= 0:
            raise ValueError("vnodes must be positive")

        self.vnodes = vnodes
        self._ring = []
        self._nodes = set()
        self._node_entries = {}

        if nodes is not None:
            for node in nodes:
                self.add(node)

    def add(self, node: str):
        if not isinstance(node, str):
            raise TypeError("node must be a str")
        if node in self._nodes:
            return

        entries = []
        for index in range(self.vnodes):
            point = self._hash(self._virtual_node_key(node, index))
            entry = (point, node)
            entries.append(entry)
            insort(self._ring, entry)

        self._nodes.add(node)
        self._node_entries[node] = entries

    def remove(self, node: str):
        if not isinstance(node, str):
            raise TypeError("node must be a str")
        if node not in self._nodes:
            return

        for entry in self._node_entries.pop(node):
            index = bisect_left(self._ring, entry)
            while index < len(self._ring) and self._ring[index] != entry:
                index += 1
            if index < len(self._ring):
                del self._ring[index]

        self._nodes.remove(node)

    def get(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        if not self._ring:
            raise LookupError("hash ring is empty")

        point = self._hash(key)
        index = bisect_left(self._ring, (point, ""))
        if index == len(self._ring):
            index = 0
        return self._ring[index][1]

    @staticmethod
    def _hash(value: str) -> int:
        return fnv1a_64(value.encode("utf-8"))

    @staticmethod
    def _virtual_node_key(node: str, index: int) -> str:
        return node + "\0" + str(index)