from collections import deque
from collections.abc import Mapping
from heapq import heappop, heappush
from itertools import count
from numbers import Number

from graphkit.graph import Graph, OrderedDict


__all__ = ["dijkstra", "shortest_path", "topo_sort"]


_NO_EDGE = object()


def _get_data(g):
    getter = getattr(g, "get_data", None)
    if callable(getter):
        data = getter()
        if data is not None:
            return data
    return g


def _adjacency_mapping(g):
    data = _get_data(g)
    if isinstance(data, Mapping):
        return data

    for name in ("adj", "_adj", "adjacency", "_adjacency", "data", "_data"):
        value = getattr(data, name, None)
        if isinstance(value, Mapping):
            return value

    return None


def _looks_like_weight(value):
    if value is None:
        return True
    if isinstance(value, Number):
        return True
    if isinstance(value, Mapping):
        return any(key in value for key in ("weight", "cost", "distance", "dist"))
    return any(hasattr(value, name) for name in ("weight", "cost", "distance", "dist"))


def _looks_like_edge_pair(item):
    if isinstance(item, (str, bytes, bytearray, Mapping)):
        return False
    if not isinstance(item, (tuple, list)):
        return False
    if len(item) != 2:
        return False
    return _looks_like_weight(item[1])


def _edge_weight(edge):
    if edge is _NO_EDGE:
        return 1

    if edge is None:
        return 1

    if isinstance(edge, Mapping):
        for key in ("weight", "cost", "distance", "dist"):
            if key in edge:
                return edge[key]
        return 1

    for name in ("weight", "cost", "distance", "dist"):
        if hasattr(edge, name):
            return getattr(edge, name)

    return edge


def _neighbor_entries(container):
    if container is None:
        return

    if isinstance(container, Mapping):
        for neighbor, edge in container.items():
            yield neighbor, edge
        return

    if isinstance(container, (str, bytes, bytearray)):
        yield container, _NO_EDGE
        return

    for item in container:
        if _looks_like_edge_pair(item):
            yield item[0], item[1]
        else:
            yield item, _NO_EDGE


def _neighbors_container(g, node):
    mapping = _adjacency_mapping(g)
    if mapping is not None:
        return mapping.get(node, ())

    for name in ("neighbors", "successors", "adjacent", "adjacent_to"):
        method = getattr(g, name, None)
        if callable(method):
            try:
                return method(node)
            except (KeyError, IndexError):
                return ()

    try:
        return g[node]
    except (TypeError, KeyError, IndexError, AttributeError):
        return ()


def _iter_edges(g, node):
    for neighbor, edge in _neighbor_entries(_neighbors_container(g, node)):
        yield neighbor, _edge_weight(edge)


def _check_non_negative(weight):
    if weight < 0:
        raise ValueError("dijkstra requires non-negative edge weights")


def _single_source(g, start, goal=None):
    counter = count()
    costs = {start: 0}
    predecessors = {start: None}
    heap = [(0, next(counter), start)]
    visited = set()

    while heap:
        current_cost, _, node = heappop(heap)

        if node in visited:
            continue

        visited.add(node)

        if goal is not None and node == goal:
            break

        for neighbor, weight in _iter_edges(g, node):
            _check_non_negative(weight)
            new_cost = current_cost + weight

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                predecessors[neighbor] = node
                if neighbor not in visited:
                    heappush(heap, (new_cost, next(counter), neighbor))

    return costs, predecessors


def dijkstra(g, start) -> dict:
    costs, _ = _single_source(g, start)
    return costs


def shortest_path(g, start, goal) -> list:
    if start == goal:
        return [start]

    costs, predecessors = _single_source(g, start, goal)

    if goal not in costs:
        return []

    path = []
    node = goal

    while True:
        path.append(node)
        if node == start:
            break
        if node not in predecessors:
            return []
        node = predecessors[node]

    path.reverse()
    return path


def _remember(ordered, node):
    if node not in ordered:
        ordered[node] = None


def _initial_nodes(g):
    seen = OrderedDict()
    mapping = _adjacency_mapping(g)

    if mapping is not None:
        for node, neighbors in mapping.items():
            _remember(seen, node)
            for neighbor, _ in _neighbor_entries(neighbors):
                _remember(seen, neighbor)
        return list(seen.keys())

    nodes_method = getattr(g, "nodes", None)
    if callable(nodes_method):
        for node in nodes_method():
            _remember(seen, node)
    else:
        try:
            for node in g:
                _remember(seen, node)
        except TypeError:
            pass

    index = 0
    nodes = list(seen.keys())
    while index < len(nodes):
        node = nodes[index]
        for neighbor, _ in _iter_edges(g, node):
            if neighbor not in seen:
                _remember(seen, neighbor)
                nodes.append(neighbor)
        index += 1

    return list(seen.keys())


def topo_sort(g) -> list:
    nodes = list(_initial_nodes(g))
    indegree = OrderedDict()
    outgoing = OrderedDict()

    for node in nodes:
        indegree[node] = 0
        outgoing[node] = []

    index = 0
    while index < len(nodes):
        node = nodes[index]
        if node not in outgoing:
            outgoing[node] = []
        if node not in indegree:
            indegree[node] = 0

        for neighbor, _ in _iter_edges(g, node):
            if neighbor not in indegree:
                indegree[neighbor] = 0
                outgoing[neighbor] = []
                nodes.append(neighbor)
            outgoing[node].append(neighbor)
            indegree[neighbor] += 1

        index += 1

    ready = deque(node for node in nodes if indegree[node] == 0)
    order = []

    while ready:
        node = ready.popleft()
        order.append(node)

        for neighbor in outgoing[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)

    if len(order) != len(indegree):
        raise ValueError("directed graph has a cycle")

    return order