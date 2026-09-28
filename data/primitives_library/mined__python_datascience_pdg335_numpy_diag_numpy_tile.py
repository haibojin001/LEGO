# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg335::numpy.diag+numpy.tile
# name: numpy_primitive
# summary: Uses numpy.diag, numpy.tile across 2 repos
# anchor_symbols: ['numpy.diag', 'numpy.tile']
# observed in 2 repos: ['graspologic-org__graspologic', 'snorkel-team__snorkel']...

# --- from graspologic-org__graspologic::tests/test_models.py::TestRDPG.setUp ---
def setUp(self) -> None:
        np.random.seed(8888)
        n_verts = 500
        point1 = np.array([0.1, 0.9])
        point2 = np.array([0.9, 0.1])
        latent1 = np.tile(point1, reps=(n_verts, 1))
        latent2 = np.tile(point2, reps=(n_verts, 1))
        latent = np.concatenate((latent1, latent2), axis=0)
        p_mat = latent @ latent.T
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat)
        self.p_mat = p_mat
        self.graph = g

# --- from snorkel-team__snorkel::snorkel/labeling/model/label_model.py::LabelModel.predict_proba ---
def predict_proba(self, L: np.ndarray) -> np.ndarray:
        r"""Return label probabilities P(Y | \lambda).

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}f

        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L, seed=123)
        >>> np.around(label_model.predict_proba(L), 1)  # doctest: +SKIP
        array([[1., 0.],
               [0., 1.],
               [0., 1.]])
        """
        L_shift = L + 1  # convert to {0, 1, ..., k}
        self._set_constants(L_shift)
        L_aug = self._get_augmented_label_matrix(L_shift)
        mu = self.mu.cpu().detach().numpy()
        jtm = np.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z

# --- from snorkel-team__snorkel::snorkel/labeling/model/label_model.py::LabelModel._get_augmented_label_matrix ---
def _get_augmented_label_matrix(
        self, L: np.ndarray, higher_order: bool = False
    ) -> np.ndarray:
        """Create augmented version of label matrix.

        In augmented version, each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        higher_order
            Whether to include higher-order correlations (e.g. LF pairs) in matrix

        Returns
        -------
        np.ndarray
            An [n,m*k] dense matrix with values in {0,1}
        """
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure
        self.c_data: Dict[int, _CliqueData] = {}
        for i in range(self.m):
            self.c_data[i] = _CliqueData(
                start_index=i * self.cardinality,
                end_index=(i + 1) * self.cardinality,
                max_cliques=set(
                    [
                        j
                        for j in self.c_tree.nodes()
                        if i in self.c_tree.nodes[j]["members"]
                    ]
                ),
            )

        L_ind = self._create_L_ind(L)

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if higher_order:
            L_aug = np.copy(L_ind)
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.nodes[item]
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                else:
                    raise ValueError(item)
                members = list(C["members"])

                # With unary maximal clique, just store its existing index
                C["start_index"] = members[0] * self.cardinality
                C["end_index"] = (members[0] + 1) * self.cardinality
            return L_aug
        else:
            return L_ind
