# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg270::numpy.sum+numpy.testing.assert_equal+numpy.testing.assert_raises
# name: numpy_pyod_primitive
# summary: Uses numpy.sum, numpy.testing.assert_equal, numpy.testing.assert_raises, pyod.utils.utility.argmaxn across 2 repos
# anchor_symbols: ['numpy.sum', 'numpy.testing.assert_equal', 'numpy.testing.assert_raises', 'pyod.utils.utility.argmaxn']
# observed in 2 repos: ['yzhao062__combo', 'yzhao062__pyod']...

# --- from yzhao062__pyod::pyod/test/test_utility.py::TestMetrics.test_argmaxn ---
def test_argmaxn(self):
        ind = argmaxn(self.value_lists, 3)
        assert_equal(len(ind), 3)

        ind = argmaxn(self.value_lists, 3)
        assert_equal(np.sum(ind), np.sum([4, 6, 9]))

        ind = argmaxn(self.value_lists, 3, order='asc')
        assert_equal(np.sum(ind), np.sum([3, 7, 8]))

        with assert_raises(ValueError):
            argmaxn(self.value_lists, -1)
        with assert_raises(ValueError):
            argmaxn(self.value_lists, 20)

# --- from yzhao062__combo::combo/test/test_utility.py::TestMetrics.test_argmaxn ---
def test_argmaxn(self):
        ind = argmaxn(self.value_lists, 3)
        assert_equal(len(ind), 3)

        ind = argmaxn(self.value_lists, 3)
        assert_equal(np.sum(ind), np.sum([4, 6, 9]))

        ind = argmaxn(self.value_lists, 3, order='asc')
        assert_equal(np.sum(ind), np.sum([3, 7, 8]))

        with assert_raises(ValueError):
            argmaxn(self.value_lists, -1)
        with assert_raises(ValueError):
            argmaxn(self.value_lists, 20)

# --- from yzhao062__pyod::pyod/test/test_base.py::TestBASE.test_init ---
def test_init(self):
        """
        Test base class initialization

        :return:
        """
        self.dummy_clf = Dummy1()
        assert_equal(self.dummy_clf.contamination, 0.1)

        self.dummy_clf = Dummy1(contamination=0.2)
        assert_equal(self.dummy_clf.contamination, 0.2)

        with assert_raises(ValueError):
            Dummy1(contamination=0.51)

        with assert_raises(ValueError):
            Dummy1(contamination=0)

        with assert_raises(ValueError):
            Dummy1(contamination=-0.5)
