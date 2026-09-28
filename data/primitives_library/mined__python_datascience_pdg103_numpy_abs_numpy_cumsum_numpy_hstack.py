# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg103::numpy.abs+numpy.cumsum+numpy.hstack
# name: numpy_typing_primitive
# summary: Uses numpy.abs, numpy.cumsum, numpy.hstack, numpy.ones across 3 repos
# anchor_symbols: ['numpy.abs', 'numpy.cumsum', 'numpy.hstack', 'numpy.ones', 'typing.cast']
# observed in 3 repos: ['capitalone__DataProfiler', 'cleanlab__cleanlab', 'piskvorky__gensim']...

# --- from piskvorky__gensim::gensim/similarities/docsim.py::_nlargest ---
def _nlargest(n, iterable):
    """Helper for extracting n documents with maximum similarity.

    Parameters
    ----------
    n : int
        Number of elements to be extracted
    iterable : iterable of list of (int, float)
        Iterable containing documents with computed similarities

    Returns
    -------
    :class:`list`
        List with the n largest elements from the dataset defined by iterable.

    Notes
    -----
    Elements are compared by the absolute value of similarity, because negative value of similarity
    does not mean some form of dissimilarity.

    """
    return heapq.nlargest(n, itertools.chain(*iterable), key=lambda item: abs(item[1]))

# --- from cleanlab__cleanlab::cleanlab/datalab/internal/issue_manager/noniid.py::NonIIDIssueManager._get_statistics ---
def _get_statistics(
        self,
        neighbor_index_distances,
    ) -> dict[str, float]:
        neighbor_index_distances = neighbor_index_distances.flatten()
        sorted_neighbors = np.sort(neighbor_index_distances)
        sorted_neighbors = np.hstack([sorted_neighbors, np.ones((1)) * (self.N - 1)]).astype(int)

        if self.background_distribution is None:
            self.background_distribution = (self.N - np.arange(1, self.N)) / (
                self.N * (self.N - 1) / 2
            )

        background_distribution = cast(np.ndarray, self.background_distribution)
        background_cdf = np.cumsum(background_distribution)

        foreground_cdf = np.arange(sorted_neighbors.shape[0]) / (sorted_neighbors.shape[0] - 1)

        statistic = np.max(np.abs(foreground_cdf - background_cdf[sorted_neighbors - 1]))
        statistics = {"ks": statistic}
        return statistics

# --- from cleanlab__cleanlab::cleanlab/datalab/internal/issue_manager/noniid.py::NonIIDIssueManager._permutation_test ---
def _permutation_test(self, num_permutations) -> float:
        N = self.N

        if self.seed is not None:
            np.random.seed(self.seed)
        perms = np.fromiter(
            itertools.chain.from_iterable(
                np.random.permutation(N) for i in range(num_permutations)
            ),
            dtype=int,
        ).reshape(num_permutations, N)

        neighbor_index_choices = self.neighbor_index_choices
        neighbor_index_choices = neighbor_index_choices.reshape(1, *neighbor_index_choices.shape)
        perm_neighbor_choices = perms[:, neighbor_index_choices].reshape(
            num_permutations, *neighbor_index_choices.shape[1:]
        )
        neighbor_index_distances = np.abs(perms[..., None] - perm_neighbor_choices).reshape(
            num_permutations, -1
        )

        statistics = []
        for neighbor_index_dist in neighbor_index_distances:
            stats = self._get_statistics(
                neighbor_index_dist,
            )
            statistics.append(stats)

        ks_stats = np.array([stats["ks"] for stats in statistics])
        ks_stats_kde = gaussian_kde(ks_stats)
        p_value = ks_stats_kde.integrate_box(self.statistics["ks"], 100)

        return p_value

# --- from piskvorky__gensim::gensim/similarities/docsim.py::Similarity.__getitem__ ---
def __getitem__(self, query):
        """Get similarities of the document (or corpus) `query` to all documents in the corpus.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            A single document in bag-of-words format, or a corpus (iterable) of such documents.

        Return
        ------
        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            Similarities of the query against this index.

        Notes
        -----
        If `query` is a corpus (iterable of documents), return a matrix of similarities of
        all query documents vs. all corpus document. This batch query is more efficient than computing the similarities
        one document after another.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim.similarities import Similarity
            >>>
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity('temp', corpus, num_features=400)
            >>> result = index[corpus]  # pairwise similarities of each document against each document

        """
        self.close_shard()  # no-op if no documents added to index since last query

        # reset num_best and normalize parameters, in case they were changed dynamically
        for shard in self.shards:
            shard.num_best = self.num_best
            shard.normalize = self.norm

        # there are 4 distinct code paths, depending on whether input `query` is
        # a corpus (or numpy/scipy matrix) or a single document, and whether the
        # similarity result should be a full array or only num_best most similar
        # documents.
        pool, shard_results = self.query_shards(query)
        if self.num_best is None:
            # user asked for all documents => just stack the sub-results into a single matrix
            # (works for both corpus / single doc query)
            result = numpy.hstack(list(shard_results))
        else:
            # the following uses a lot of lazy evaluation and (optionally) parallel
            # processing, to improve query latency and minimize memory footprint.
            offsets = numpy.cumsum([0] + [len(shard) for shard in self.shards])

            def convert(shard_no, doc):
                return [(doc_index + offsets[shard_no], sim) for doc_index, sim in doc]

            is_corpus, query = utils.is_corpus(query)
            is_corpus = is_corpus or hasattr(query, 'ndim') and query.ndim > 1 and query.shape[0] > 1
            if not is_corpus:
                # user asked for num_best most similar and query is a single doc
                results = (convert(shard_no, result) for shard_no, result in enumerate(shard_results))
                result = _nlargest(self.num_best, results)
            else:
                # the trickiest combination: returning num_best results when query was a corpus
                results = []
                for shard_no, result in enumerate(shard_results):
                    shard_result = [convert(shard_no, doc) for doc in result]
                    results.append(shard_result)
                result = []
                for parts in zip(*results):
                    merged = _nlargest(self.num_best, parts)
                    result.append(merged)
        if pool:
            # gc doesn't seem to collect the Pools, eventually leading to
            # "IOError 24: too many open files". so let's terminate it manually.
            pool.terminate()

        return result

# --- from capitalone__DataProfiler::dataprofiler/labelers/labeler_utils.py::evaluate_accuracy ---
def evaluate_accuracy(
    predicted_entities_in_index: list[list[int]],
    true_entities_in_index: list[list[int]],
    num_labels: int,
    entity_rev_dict: dict[int, str],
    verbose: bool = True,
    omitted_labels: tuple[str, ...] = ("PAD", "UNKNOWN"),
    confusion_matrix_file: str | None = None,
) -> tuple[float, dict]:
    """
    Evaluate accuracy from comparing predicted labels with true labels.

    :param predicted_entities_in_index: predicted encoded labels for input
        sentences
    :type predicted_entities_in_index: list(array(int))
    :param true_entities_in_index: true encoded labels for input sentences
    :type true_entities_in_index: list(array(int))
    :param entity_rev_dict: dictionary to convert indices to entities
    :type entity_rev_dict: dict([index, entity])
    :param verbose: print additional information for debugging
    :type verbose: boolean
    :param omitted_labels: labels to omit from the accuracy evaluation
    :type omitted_labels: list() of text labels
    :param confusion_matrix_file: File name (and dir) for confusion matrix
    :type confusion_matrix_file: str
    :return : f1-score
    :rtype: float
    """
    label_names = None
    label_indexes = None
    if entity_rev_dict:
        label_names = [
            str(x[1])
            for x in sorted(entity_rev_dict.items(), key=lambda x: x[0])
            if x[1] not in omitted_labels
        ]
        label_indexes = [
            x[0]
            for x in sorted(entity_rev_dict.items(), key=lambda x: x[0])
            if x[1] not in omitted_labels
        ]

    max_len = len(predicted_entities_in_index[0])
    true_labels_padded = np.zeros((len(true_entities_in_index), max_len))
    for i, true_labels_row in enumerate(true_entities_in_index):
        true_labels_padded[i][: len(true_labels_row)] = true_labels_row

    true_labels_flatten = np.hstack(true_labels_padded)  # type: ignore
    predicted_labels_flatten = np.hstack(predicted_entities_in_index)

    all_labels: list[str] = []
    if entity_rev_dict:
        all_labels = [entity_rev_dict[key] for key in sorted(entity_rev_dict.keys())]

    # From sklearn, description of the confusion matrix:
    # By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    # is equal to the number of observations known to be in group :math:`i` but
    # predicted to be in group :math:`j`.
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.int64)
    batch_size = min(2**20, len(true_labels_flatten))
    for batch_ind in range(len(true_labels_flatten) // batch_size + 1):
        true_label_batch = true_labels_flatten[
            batch_size * batch_ind : (batch_ind + 1) * batch_size
        ]
        pred_label_batch = predicted_labels_flatten[
            batch_size * batch_ind : (batch_ind + 1) * batch_size
        ]
        conf_mat += scipy.sparse.coo_matrix(
            (np.ones((len(pred_label_batch),)), (true_label_batch, pred_label_batch)),
            shape=(num_labels, num_labels),
            dtype=np.int64,
        ).toarray()

    # Only write confusion matrix if file exists
    if confusion_matrix_file and entity_rev_dict:
        import pandas as pd

        conf_mat_pd = pd.DataFrame(
            conf_mat,
            columns=list(map(lambda x: "pred:" + x, all_labels)),
            index=list(map(lambda x: "true:" + x, all_labels)),
        )

        # Make directory, if required
        if os.path.dirname(confusion_matrix_file) and not os.path.isdir(
            os.path.dirname(confusion_matrix_file)
        ):
            os.makedirs(os.path.dirname(confusion_matrix_file))

        conf_mat_pd.to_csv(confusion_matrix_file)

    f1_report: dict = cast(
        Dict,
        classification_report(
            conf_mat, labels=label_indexes, target_names=label_names, output_dict=True
        ),
    )

    # adjust macro average to be updated only on positive support labels
    # note: in sklearn, support is number of occurrences of each label in
    # true_labels_flatten
    num_labels_with_positive_support = 0
    for key, values in f1_report.items():
        if key not in ["accuracy", "macro avg", "weighted avg", "micro avg"]:
            if values["support"]:
                num_labels_with_positive_support += 1

    # bc sklearn does not remove 0.0 f1 score for 0 support in macro avg.
    for metric in f1_report["macro avg"].keys():
        if metric != "support":
            if not num_labels_with_positive_support:
                f1_report["macro avg"][metric] = np.nan
            else:
                if not label_names:
                    f1_report["macro avg"][metric] = 0
                else:
                    f1_report["macro avg"][metric] *= (
                        float(len(label_names)) / num_labels_with_positive_support
                    )

    if "macro avg" in f1_report:
        f1: float = f1_report["macro avg"]["f1-score"]  # this is micro for the report
    else:
        # this is the only remaining option for the report
        f1 = f1_report["accuracy"]

    if verbose:
        if not label_names:
            label_names = [""]

        f1_report_str = f1_report_dict_to_str(f1_report, label_names)
        logger.info(f"(After removing non-entity tokens)\n{f1_report_str}")
        logger.info(f"F1 Score: {f1}")

    return f1, f1_report
