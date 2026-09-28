"""Traversal and connectivity helpers for graphkit."""

from collections import deque
from numbers import Number

from graphkit.graph import Graph, OrderedDict

__all__ = ["bfs", "dfs", "connected_components"]


_NEIGHBOR_METHOD_NAMES = (
    "neighbors",
    "out_neighbors",
    "out_nbrs",
    "successors",
    "adjacent",
    "get_neighbors",
)

_INCOMING_METHOD_NAMES = (
    "in_neighbors",
    "in_nbrs",
    "inc_nbrs",
    "predecessors",
)

_ADJACENCY_ATTR_NAMES = (
    "adj",
    "_adj",
    "adjacency",
    "_adjacency",
    "succ",
    "_succ",
)

_NODE_ATTR_NAMES = (
    "_nodes",
)


def bfs(g, start) -> list:
    """Return nodes in breadth-first visitation order from *start*."""
    visited = OrderedDict()
    order = []
    queue = deque([start])
    visited[start] = None

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in _neighbors(g, node):
            if neighbor not in visited:
                visited[neighbor] = None
                queue.append(neighbor)

    return order


def dfs(g, start) -> list:
    """Return nodes in depth-first pre-order from *start*.

    Neighbors are considered in their insertion order.
    """
    visited = OrderedDict()
    order = []
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited[node] = None
        order.append(node)

        neighbors = _neighbors(g, node)
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


def connected_components(g) -> list:
    """Return connected components of an undirected graph.

    The result is a list of components. Each component is returned as a sorted
    list of nodes. Components are discovered in the graph's node insertion
    order where that order is available.
    """
    nodes = _all_nodes(g)
    visited = OrderedDict()
    components = []

    for start in nodes:
        if start in visited:
            continue

        component = []
        queue = deque([start])
        visited[start] = None

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor in _undirected_neighbors(g, node, nodes):
                if neighbor not in visited:
                    visited[neighbor] = None
                    queue.append(neighbor)

        components.append(_sorted_nodes(component))

    return components


def _neighbors(g, node):
    for name in _NEIGHBOR_METHOD_NAMES:
        method = getattr(g, name, None)
        if callable(method):
            try:
                return _neighbor_nodes(method(node))
            except (KeyError, IndexError):
                return []
            except TypeError:
                continue

    adjacency = _adjacency_mapping(g)
    if adjacency is not None:
        try:
            if node in adjacency:
                return _neighbor_nodes(adjacency[node])
        except TypeError:
            pass

        get = getattr(adjacency, "get", None)
        if callable(get):
            return _neighbor_nodes(get(node, ()))

    if hasattr(g, "get") and callable(g.get):
        try:
            return _neighbor_nodes(g.get(node, ()))
        except TypeError:
            pass

    try:
        return _neighbor_nodes(g[node])
    except (KeyError, IndexError, TypeError, AttributeError):
        return []


def _incoming_neighbors(g, node, nodes=None):
    for name in _INCOMING_METHOD_NAMES:
        method = getattr(g, name, None)
        if callable(method):
            try:
                return _neighbor_nodes(method(node))
            except (KeyError, IndexError):
                return []
            except TypeError:
                continue

    if nodes is None:
        nodes = _all_nodes(g)

    incoming = []
    for other in nodes:
        if other == node:
            continue
        for neighbor in _neighbors(g, other):
            if neighbor == node:
                incoming.append(other)
                break

    return incoming


def _undirected_neighbors(g, node, nodes=None):
    seen = OrderedDict()

    for neighbor in _neighbors(g, node):
        if neighbor not in seen:
            seen[neighbor] = None

    for neighbor in _incoming_neighbors(g, node, nodes):
        if neighbor not in seen:
            seen[neighbor] = None

    return list(seen.keys())


def _all_nodes(g):
    nodes = OrderedDict()

    def add(node):
        if node not in nodes:
            nodes[node] = None

    node_source_found = False

    node_method = getattr(g, "nodes", None)
    if callable(node_method):
        try:
            for node in node_method():
                add(node)
            node_source_found = True
        except TypeError:
            pass
    elif node_method is not None:
        _add_node_source(nodes, node_method)
        node_source_found = True

    for name in _NODE_ATTR_NAMES:
        source = getattr(g, name, None)
        if source is not None:
            _add_node_source(nodes, source)
            node_source_found = True

    adjacency = _adjacency_mapping(g)
    if adjacency is not None:
        _add_adjacency_nodes(nodes, adjacency)
        node_source_found = True

    if _looks_like_mapping(g):
        _add_adjacency_nodes(nodes, g)
        node_source_found = True

    if not node_source_found:
        try:
            for node in g:
                add(node)
        except TypeError:
            pass

    return list(nodes.keys())


def _add_node_source(nodes, source):
    if _looks_like_mapping(source):
        iterable = source.keys()
    else:
        iterable = _as_list(source)

    for node in iterable:
        if node not in nodes:
            nodes[node] = None


def _add_adjacency_nodes(nodes, adjacency):
    try:
        items = adjacency.items()
    except AttributeError:
        return

    for node, neighbors in items:
        if node not in nodes:
            nodes[node] = None
        for neighbor in _neighbor_nodes(neighbors):
            if neighbor not in nodes:
                nodes[neighbor] = None


def _adjacency_mapping(g):
    for name in _ADJACENCY_ATTR_NAMES:
        value = getattr(g, name, None)
        if value is not None and _looks_like_mapping(value):
            return value

    if _looks_like_mapping(g):
        return g

    return None


def _looks_like_mapping(value):
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _as_list(value):
    if value is None:
        return []

    if _looks_like_mapping(value):
        return list(value.keys())

    if isinstance(value, (str, bytes)):
        return [value]

    try:
        return list(value)
    except TypeError:
        return [value]


def _neighbor_nodes(value):
    if value is None:
        return []

    if _looks_like_mapping(value):
        return list(value.keys())

    if isinstance(value, (str, bytes)):
        return [value]

    try:
        entries = list(value)
    except TypeError:
        entries = [value]

    return [_neighbor_from_entry(entry) for entry in entries]


def _neighbor_from_entry(entry):
    if _looks_like_weighted_neighbor_entry(entry):
        return entry[0]
    return entry


def _looks_like_weighted_neighbor_entry(entry):
    if isinstance(entry, (str, bytes)):
        return False

    if not isinstance(entry, (tuple, list)):
        return False

    if len(entry) != 2:
        return False

    return isinstance(entry[1], Number)


def _sorted_nodes(nodes):
    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=lambda item: (type(item).__name__, repr(item)))