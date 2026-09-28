# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg515::matplotlib.pyplot.subplots+sklearn.linear_model.LogisticRegression
# name: matplotlib_sklearn_primitive
# summary: Uses matplotlib.pyplot.subplots, sklearn.linear_model.LogisticRegression across 2 repos
# anchor_symbols: ['matplotlib.pyplot.subplots', 'sklearn.linear_model.LogisticRegression']
# observed in 2 repos: ['ploomber__sklearn-evaluation', 'reiinakano__scikit-plot']...

# --- from ploomber__sklearn-evaluation::tests/test_cumulative_gain_lift_curve.py::test_ax_lift_curve ---
def test_ax_lift_curve():
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    fig, ax = plt.subplots(1, 1)
    out_ax = lift_curve(y, probas)
    assert ax is not out_ax
    out_ax = lift_curve(y, probas, ax=ax)
    assert ax is out_ax

# --- from ploomber__sklearn-evaluation::tests/test_cumulative_gain_lift_curve.py::test_ax_cumulative_gain ---
def test_ax_cumulative_gain():
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    fig, ax = plt.subplots(1, 1)
    out_ax = cumulative_gain(y, probas)
    assert ax is not out_ax
    out_ax = cumulative_gain(y, probas, ax=ax)
    assert ax is out_ax

# --- from reiinakano__scikit-plot::scikitplot/tests/test_estimators.py::TestPlotLearningCurve.test_ax ---
def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_learning_curve(clf, self.X, self.y)
        assert ax is not out_ax
        out_ax = plot_learning_curve(clf, self.X, self.y, ax=ax)
        assert ax is out_ax

# --- from reiinakano__scikit-plot::scikitplot/tests/test_classifiers.py::TestPlotROCCurve.test_ax ---
def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_roc_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_roc_curve(self.X, self.y, ax=ax)
        assert ax is out_ax
