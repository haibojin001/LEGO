# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg484::networkx.DiGraph+networkx.Graph
# name: networkx_primitive
# summary: Uses networkx.DiGraph, networkx.Graph across 2 repos
# anchor_symbols: ['networkx.DiGraph', 'networkx.Graph']
# observed in 2 repos: ['graspologic-org__graspologic', 'stellargraph__stellargraph']...

# --- from graspologic-org__graspologic::graspologic/pipeline/graph_builder.py::GraphBuilder.__init__ ---
def __init__(self, directed: bool = False):
        # OrderedDict is the default for {} anyway, but I wanted to be very explicit,
        # since we absolutely rely on the ordering
        self._id_map: Dict[Any, int] = OrderedDict()
        self._graph = nx.DiGraph() if directed else nx.Graph()

# --- from stellargraph__stellargraph::tests/test_utils/graphs.py::example_graph_nx ---
def example_graph_nx(
    feature_size=None, label="default", feature_name="feature", is_directed=False
):
    graph = nx.DiGraph() if is_directed else nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
    graph.add_nodes_from([1, 2, 3, 4], label=label)
    graph.add_edges_from(elist, label=label)

    # Add example features
    if feature_size is not None:
        for v in graph.nodes():
            graph.nodes[v][feature_name] = int(v) * np.ones(feature_size)

    return graph

# --- from graspologic-org__graspologic::graspologic/models/edge_swaps.py::EdgeSwapper._do_setup ---
def _do_setup(self) -> np.ndarray:
        """
        Computes the edge_list from the adjancency matrix

        Returns
        -------
        edge_list : np.ndarray, shape (n_verts, 2)
            The corresponding edge_list of adjacency
        """

        # get edges for upper triangle of undirected graph
        row_inds, col_inds = np.nonzero(self.adjacency)
        upper = row_inds < col_inds
        row_inds = row_inds[upper]
        col_inds = col_inds[upper]
        edge_list = np.stack((row_inds, col_inds)).T
        return edge_list

# --- from stellargraph__stellargraph::tests/core/test_graph.py::test_from_networkx_empty ---
def test_from_networkx_empty():
    empty = StellarGraph.from_networkx(nx.Graph())
    assert not empty.is_directed()
    assert empty.node_types == set()
    assert isinstance(empty, StellarGraph)

    empty = StellarGraph.from_networkx(nx.DiGraph())
    assert empty.is_directed()
    assert empty.node_types == set()
    assert isinstance(empty, StellarDiGraph)

    # https://github.com/stellargraph/stellargraph/issues/1339
    features = pd.DataFrame(columns=range(10))
    empty_with_features = StellarGraph.from_networkx(nx.Graph(), node_features=features)
    assert empty_with_features.node_types == set()
