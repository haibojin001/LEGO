from collections.abc import MutableSet


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        marker = [None]
        marker.append(marker)
        marker.append(marker)
        self.end = marker
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key in self.map:
            return
        marker = self.end
        previous = marker[1]
        node = [key, previous, marker]
        previous[2] = node
        marker[1] = node
        self.map[key] = node

    def discard(self, key):
        node = self.map.pop(key, None)
        if node is not None:
            previous = node[1]
            following = node[2]
            previous[2] = following
            following[1] = previous

    def __iter__(self):
        marker = self.end
        node = marker[2]
        while node is not marker:
            yield node[0]
            node = node[2]

    def __reversed__(self):
        marker = self.end
        node = marker[1]
        while node is not marker:
            yield node[0]
            node = node[1]

    def pop(self, last=True):
        if not self:
            raise KeyError("set is empty")
        node = self.end[1] if last else self.end[2]
        key = node[0]
        self.discard(key)
        return key

    def __repr__(self):
        name = self.__class__.__name__
        if not self:
            return f"{name}()"
        return f"{name}({list(self)!r})"

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


if __name__ == "__main__":
    s = OrderedSet("abracadaba")
    t = OrderedSet("simsalabim")
    print(s | t)
    print(s & t)
    print(s - t)