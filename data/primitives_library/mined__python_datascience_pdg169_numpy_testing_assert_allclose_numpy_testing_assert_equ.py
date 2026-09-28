# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg169::numpy.testing.assert_allclose+numpy.testing.assert_equal
# name: numpy_primitive
# summary: Uses numpy.testing.assert_allclose, numpy.testing.assert_equal across 4 repos
# anchor_symbols: ['numpy.testing.assert_allclose', 'numpy.testing.assert_equal']
# observed in 4 repos: ['CamDavidsonPilon__lifetimes', 'mwaskom__seaborn', 'yzhao062__combo', 'yzhao062__pyod']...

# --- from yzhao062__pyod::pyod/test/test_ecod_parallel.py::TestCOPODParallel.test_train_scores ---
def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])
        assert_allclose(self.clf.decision_scores_, self.clf_.decision_scores_)

# --- from yzhao062__pyod::pyod/test/test_ecod.py::TestECODParallel.test_train_scores ---
def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])
        assert_allclose(self.clf.decision_scores_, self.clf_.decision_scores_)

# --- from mwaskom__seaborn::tests/test_regression.py::TestRegressionPlotter.test_numeric_bins ---
def test_numeric_bins(self):

        p = lm._RegressionPlotter(self.df.x, self.df.y)
        x_binned, bins = p.bin_predictor(self.bins_numeric)
        npt.assert_equal(len(bins), self.bins_numeric)
        npt.assert_array_equal(np.unique(x_binned), bins)

# --- from mwaskom__seaborn::seaborn/regression.py::_RegressionPlotter.bin_predictor ---
def bin_predictor(self, bins):
        """Discretize a predictor by assigning value to closest bin."""
        x = np.asarray(self.x)
        if np.isscalar(bins):
            percentiles = np.linspace(0, 100, bins + 2)[1:-1]
            bins = np.percentile(x, percentiles)
        else:
            bins = np.ravel(bins)

        dist = np.abs(np.subtract.outer(x, bins))
        x_binned = bins[np.argmin(dist, axis=1)].ravel()

        return x_binned, bins

# --- from CamDavidsonPilon__lifetimes::tests/test_plotting.py::TestPlotting.test_plot_period_transactions_mbgf ---
def test_plot_period_transactions_mbgf(self, cd_data):

        mbgf = ModifiedBetaGeoFitter()
        mbgf.fit(cd_data["frequency"], cd_data["recency"], cd_data["T"])

        ax = plotting.plot_period_transactions(mbgf)

        assert_equal(ax.title.get_text(), "Frequency of Repeat Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "Number of Calibration Period Transactions")
        assert_equal(ax.yaxis.get_label().get_text(), "Customers")
        assert_array_equal([label.get_text() for label in ax.legend_.get_texts()], ["Actual", "Model"])
        plt.close()

# --- from CamDavidsonPilon__lifetimes::tests/test_plotting.py::TestPlotting.test_plot_period_transactions ---
def test_plot_period_transactions(self, bgf):
        expected = [1411, 439, 214, 100, 62, 38, 29, 1411, 439, 214, 100, 62, 38, 29]

        ax = plotting.plot_period_transactions(bgf)

        assert_allclose([p.get_height() for p in ax.patches], expected, rtol=0.3)
        assert_equal(ax.title.get_text(), "Frequency of Repeat Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "Number of Calibration Period Transactions")
        assert_equal(ax.yaxis.get_label().get_text(), "Customers")
        assert_array_equal([label.get_text() for label in ax.legend_.get_texts()], ["Actual", "Model"])
        plt.close()

# --- from yzhao062__combo::combo/test/test_utility.py::TestScaler.test_normalization ---
def test_normalization(self):

        # test when X_t is presented and no scalar
        norm_X_train, norm_X_test = standardizer(self.X_train, self.X_test)
        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        assert_allclose(norm_X_test.mean(), 0, atol=0.05)
        assert_allclose(norm_X_test.std(), 1, atol=0.05)

        # test when X_t is not presented and no scalar
        norm_X_train = standardizer(self.X_train)
        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        # test when X_t is presented and the scalar is kept
        norm_X_train, norm_X_test, scalar = standardizer(self.X_train,
                                                         self.X_test,
                                                         keep_scalar=True)

        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        assert_allclose(norm_X_test.mean(), 0, atol=0.05)
        assert_allclose(norm_X_test.std(), 1, atol=0.05)

        if not hasattr(scalar, 'fit') or not hasattr(scalar, 'transform'):
            raise AttributeError("%s is not a detector instance." % (scalar))

        # test when X_t is not presented and the scalar is kept
        norm_X_train, scalar = standardizer(self.X_train, keep_scalar=True)

        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        if not hasattr(scalar, 'fit') or not hasattr(scalar, 'transform'):
            raise AttributeError("%s is not a detector instance." % (scalar))

        # test shape difference
        with assert_raises(ValueError):
            standardizer(self.X_train, self.X_test_diff)
