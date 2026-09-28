# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg101::numpy.count_nonzero+numpy.issubdtype
# name: numpy_primitive
# summary: Uses numpy.count_nonzero, numpy.issubdtype across 3 repos
# anchor_symbols: ['numpy.count_nonzero', 'numpy.issubdtype']
# observed in 3 repos: ['graspologic-org__graspologic', 'piskvorky__gensim', 'vi3k6i5__GuidedLDA']...

# --- from graspologic-org__graspologic::tests/test_sims.py::TestMMSBM.test_labels ---
def test_labels(self):
        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)
        alpha = np.array([0.0001, 100])  # assign all nodes to second group(index 1)
        A = mmsbm(
            self.n,
            self.p,
            alpha,
            rng=rng,
            directed=False,
            loops=False,
            return_labels=True,
        )
        labels = A[1]
        expected_labels = np.ones((self.n, self.n))
        np.fill_diagonal(expected_labels, -1)
        # since we expect all nodes on the off diagonal to pertain to community 2
        # we expect all values on the off diagonal to be 1
        # We expect nan on the diagonal as this graph has no loops

        # check that expected labels is the same as the labels output by function
        self.assertTrue(np.allclose(labels, expected_labels))
        pass

# --- from vi3k6i5__GuidedLDA::guidedlda/utils.py::dtm2ldac ---
def dtm2ldac(dtm, offset=0):
    """Convert a document-term matrix into an LDA-C formatted file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    doclines : iterable of LDA-C lines suitable for writing to file

    Notes
    -----
    If a format similar to SVMLight is desired, `offset` of 1 may be used.
    """
    try:
        dtm = dtm.tocsr()
    except AttributeError:
        pass
    assert np.issubdtype(dtm.dtype, int)
    n_rows = dtm.shape[0]
    for i, row in enumerate(dtm):
        try:
            row = row.toarray().squeeze()
        except AttributeError:
            pass
        unique_terms = np.count_nonzero(row)
        if unique_terms == 0:
            raise ValueError("dtm row {} has all zero entries.".format(i))
        term_cnt_pairs = [(i + offset, cnt) for i, cnt in enumerate(row) if cnt > 0]
        docline = str(unique_terms) + ' '
        docline += ' '.join(["{}:{}".format(i, cnt) for i, cnt in term_cnt_pairs])
        if (i + 1) % 1000 == 0:
            logger.info("dtm2ldac: on row {} of {}".format(i + 1, n_rows))
        yield docline

# --- from graspologic-org__graspologic::graspologic/simulations/simulations.py::sample_edges ---
def sample_edges(
    P: np.ndarray, directed: bool = False, loops: bool = False
) -> np.ndarray:
    """
    Gemerates a binary random graph based on the P matrix provided

    Each element in P represents the probability of a connection between
    a vertex indexed by the row i and the column j.

    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for a random graph
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    Returns
    -------
    A: ndarray (n_vertices, n_vertices)
        Binary adjacency matrix the same size as P representing a random
        graph

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    """
    if type(P) is not np.ndarray:
        raise TypeError("P must be numpy.ndarray")
    if len(P.shape) != 2:
        raise ValueError("P must have dimension 2 (n_vertices, n_dimensions)")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(P.shape[0])
        samples = np.random.binomial(1, P[triu_inds])
        A = np.zeros_like(P)
        A[triu_inds] = samples
        A = symmetrize(A, method="triu")
    else:
        A = np.random.binomial(1, P)

    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))

# --- from piskvorky__gensim::gensim/matutils.py::unitvec ---
def unitvec(vec, norm='l2', return_norm=False):
    """Scale a vector to unit length.

    Parameters
    ----------
    vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
        Input vector in any format
    norm : {'l1', 'l2', 'unique'}, optional
        Metric to normalize in.
    return_norm : bool, optional
        Return the length of vector `vec`, in addition to the normalized vector itself?

    Returns
    -------
    numpy.ndarray, scipy.sparse, list of (int, float)}
        Normalized vector in same format as `vec`.
    float
        Length of `vec` before normalization, if `return_norm` is set.

    Notes
    -----
    Zero-vector will be unchanged.

    """
    supported_norms = ('l1', 'l2', 'unique')
    if norm not in supported_norms:
        raise ValueError("'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))

    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if norm == 'unique':
            veclen = vec.nnz
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            vec /= veclen
            if return_norm:
                return vec, veclen
            else:
                return vec
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            if vec.size == 0:
                veclen = 0.0
            else:
                veclen = blas_nrm2(vec)
        if norm == 'unique':
            veclen = np.count_nonzero(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            if return_norm:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    try:
        first = next(iter(vec))  # is there at least one element?
    except StopIteration:
        if return_norm:
            return vec, 1.0
        else:
            return vec

    if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        if norm == 'unique':
            length = 1.0 * len(vec)
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        if return_norm:
            return ret_normalized_vec(vec, length), length
        else:
            return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")
