# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg271::numpy.testing.assert_equal+sklearn.metrics.precision_score
# name: numpy_sklearn_primitive
# summary: Uses numpy.testing.assert_equal, sklearn.metrics.precision_score across 2 repos
# anchor_symbols: ['numpy.testing.assert_equal', 'sklearn.metrics.precision_score']
# observed in 2 repos: ['yzhao062__combo', 'yzhao062__pyod']...

# --- from yzhao062__pyod::pyod/test/test_utility.py::TestMetrics.test_precision_n_scores ---
def test_precision_n_scores(self):
        assert_equal(precision_score(self.y, self.manual_labels),
                     precision_n_scores(self.y, self.labels_))

# --- from yzhao062__combo::combo/test/test_utility.py::TestMetrics.test_precision_n_scores ---
def test_precision_n_scores(self):
        assert_equal(precision_score(self.y, self.manual_labels),
                     precision_n_scores(self.y, self.labels_))
