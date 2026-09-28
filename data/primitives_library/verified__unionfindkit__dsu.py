class UnionFind:
    """Disjoint-set union (union-find) with path compression and union by rank."""

    def __init__(self, n: int):
        """Create n singleton sets containing elements 0 through n - 1."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self._parent = list(range(n))
        self._rank = [0] * n
        self._count = n

    def find(self, x: int) -> int:
        """Return the representative of the set containing x."""
        if x < 0 or x >= len(self._parent):
            raise IndexError("element index out of range")

        root = x
        while self._parent[root] != root:
            root = self._parent[root]

        while self._parent[x] != x:
            parent = self._parent[x]
            self._parent[x] = root
            x = parent

        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the sets containing a and b; return False if already connected."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self._rank[root_a] < self._rank[root_b]:
            self._parent[root_a] = root_b
        elif self._rank[root_a] > self._rank[root_b]:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1

        self._count -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """Return True if a and b are in the same set."""
        return self.find(a) == self.find(b)

    def count(self) -> int:
        """Return the current number of disjoint sets."""
        return self._count