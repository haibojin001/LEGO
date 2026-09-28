# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg100::numpy.column_stack+numpy.empty
# name: numpy_primitive
# summary: Uses numpy.column_stack, numpy.empty across 2 repos
# anchor_symbols: ['numpy.column_stack', 'numpy.empty']
# observed in 2 repos: ['graspologic-org__graspologic', 'piskvorky__gensim']...

# --- from graspologic-org__graspologic::graspologic/match/solver.py::_GraphMatchSolver.finalize ---
def finalize(self, P: np.ndarray, rng: np.random.Generator) -> None:
        self.convex_solution_ = P

        # project back onto the feasible region (permutations)
        if P.shape != (0, 0):
            permutation = self.linear_sum_assignment(P, rng, maximize=True)
        else:  # the case where input was all seeded
            permutation = np.array([], dtype=int)

        # deal with seed-nonseed sorting from the initialization
        permutation = np.concatenate((
            np.arange(self.n_seeds),
            permutation + self.n_seeds,
        ))
        final_permutation = np.empty(self.n, dtype=int)
        final_permutation[self.perm_A] = self.perm_B[permutation]

        # deal with un-padding
        matching = np.column_stack((np.arange(self.n), final_permutation))
        if self.padded:
            if self._padded_B:
                matching = matching[matching[:, 1] < self.n_B]
            else:
                matching = matching[: self.n_A]

        self.matching_ = matching

        # compute the objective function value for evaluation
        score = self.compute_score(final_permutation)
        self.score_ = score

# --- from piskvorky__gensim::gensim/matutils.py::corpus2dense ---
def corpus2dense(corpus, num_terms, num_docs=None, dtype=np.float32):
    """Convert corpus into a dense numpy 2D array, with documents as columns.

    Parameters
    ----------
    corpus : iterable of iterable of (int, number)
        Input corpus in the Gensim bag-of-words format.
    num_terms : int
        Number of terms in the dictionary. X-axis of the resulting matrix.
    num_docs : int, optional
        Number of documents in the corpus. If provided, a slightly more memory-efficient code path is taken.
        Y-axis of the resulting matrix.
    dtype : data-type, optional
        Data type of the output matrix.

    Returns
    -------
    numpy.ndarray
        Dense 2D array that presents `corpus`.

    See Also
    --------
    :class:`~gensim.matutils.Dense2Corpus`
        Convert dense matrix to Gensim corpus format.

    """
    if num_docs is not None:
        # we know the number of documents => don't bother column_stacking
        docno, result = -1, np.empty((num_terms, num_docs), dtype=dtype)
        for docno, doc in enumerate(corpus):
            result[:, docno] = sparse2full(doc, num_terms)
        assert docno + 1 == num_docs
    else:
        # The below used to be a generator, but NumPy deprecated generator as of 1.16 with:
        # """
        # FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple.
        # Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error
        # in the future.
        # """
        result = np.column_stack([sparse2full(doc, num_terms) for doc in corpus])
    return result.astype(dtype)
