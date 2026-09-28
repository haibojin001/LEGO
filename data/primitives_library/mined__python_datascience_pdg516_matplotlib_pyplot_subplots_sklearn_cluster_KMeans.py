# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg516::matplotlib.pyplot.subplots+sklearn.cluster.KMeans
# name: matplotlib_sklearn_primitive
# summary: Uses matplotlib.pyplot.subplots, sklearn.cluster.KMeans across 2 repos
# anchor_symbols: ['matplotlib.pyplot.subplots', 'sklearn.cluster.KMeans']
# observed in 2 repos: ['ploomber__sklearn-evaluation', 'reiinakano__scikit-plot']...

# --- from ploomber__sklearn-evaluation::tests/test_clustering.py::test_ax_elbow ---
def test_ax_elbow():
    clf = KMeans()
    fig, ax = plt.subplots(1, 1)
    out_ax = plot.elbow_curve(X, clf, ax=ax)
    assert ax is out_ax

# --- from reiinakano__scikit-plot::scikitplot/tests/test_cluster.py::TestPlotElbow.test_ax ---
def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_elbow_curve(clf, self.X)
        assert ax is not out_ax
        out_ax = plot_elbow_curve(clf, self.X, ax=ax)
        assert ax is out_ax

# --- from reiinakano__scikit-plot::scikitplot/tests/test_clustering.py::TestPlotSilhouette.test_ax ---
def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        scikitplot.clustering_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_silhouette(self.X)
        assert ax is not out_ax
        out_ax = clf.plot_silhouette(self.X, ax=ax)
        assert ax is out_ax

# --- from ploomber__sklearn-evaluation::tests/test_clustering.py::test_ax_silhouette ---
def test_ax_silhouette():
    clf = KMeans()
    ax = []
    for i in range(5):
        fig, axes = plt.subplots(1, 1)
        ax.append(axes)
    out_ax = plot.silhouette_analysis(X, clf)
    for axes in ax:
        assert axes is not out_ax
    out_ax = plot.silhouette_analysis(X, clf, ax=ax)
    assert ax[-1] is out_ax
