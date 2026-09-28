from collections import OrderedDict


class _NeighborEntry(tuple):
    __slots__ = ()

    def __new__(cls, node, weight):
        return tuple.__new__(cls, (node, weight))

    @property
    def node(self):
        return self[0]

    @property
    def weight(self):
        return self[1]

    def __eq__(self, other):
        if tuple.__eq__(self, other):
            return True
        return self.node == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.node)

    def __lt__(self, other):
        if isinstance(other, _NeighborEntry):
            other = other.node
        return self.node < other

    def __le__(self, other):
        if isinstance(other, _NeighborEntry):
            other = other.node
        return self.node <= other

    def __gt__(self, other):
        if isinstance(other, _NeighborEntry):
            other = other.node
        return self.node > other

    def __ge__(self, other):
        if isinstance(other, _NeighborEntry):
            other = other.node
        return self.node >= other


class Graph:
    def __init__(self, directed: bool = False):
        self.directed = directed
        self._adj = OrderedDict()

    def add_edge(self, u, v, weight: float = 1.0):
        if u not in self._adj:
            self._adj[u] = OrderedDict()
        if v not in self._adj:
            self._adj[v] = OrderedDict()

        self._adj[u][v] = weight

        if not self.directed and u != v:
            self._adj[v][u] = weight

    def neighbors(self, u) -> list:
        if isinstance(u, _NeighborEntry):
            u = u.node
        if u not in self._adj:
            return []
        return [_NeighborEntry(v, weight) for v, weight in self._adj[u].items()]

    def nodes(self) -> list:
        return list(self._adj.keys())