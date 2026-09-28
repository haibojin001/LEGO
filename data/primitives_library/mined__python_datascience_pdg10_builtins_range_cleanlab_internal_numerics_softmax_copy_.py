# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg10::builtins.range+cleanlab.internal.numerics.softmax+copy.copy
# name: builtins_cleanlab_primitive
# summary: Uses builtins.range, cleanlab.internal.numerics.softmax, copy.copy, evalml.exceptions.PartialDependenceError across 22 repos
# anchor_symbols: ['builtins.range', 'cleanlab.internal.numerics.softmax', 'copy.copy', 'evalml.exceptions.PartialDependenceError', 'evalml.model_understanding._partial_dependence_utils._get_feature_names_from_str_or_col_index', 'evalml.model_understanding._partial_dependence_utils._is_feature_of_semantic_type']
# observed in 22 repos: ['CamDavidsonPilon__lifelines', 'Data-Centric-AI-Community__fg-data-profiling', 'ModelOriented__DALEX', 'NannyML__nannyml', 'alteryx__evalml']...

# --- from lazyprogrammer__machine_learning_examples::rl2/cartpole/pg_theano.py::ValueModel.partial_fit ---
def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    self.train_op(X, Y)

# --- from lazyprogrammer__machine_learning_examples::rl2/mountaincar/pg_theano.py::ValueModel.partial_fit ---
def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    Y = np.atleast_1d(Y)
    self.train_op(X, Y)

# --- from mwaskom__seaborn::seaborn/_core/scales.py::ContinuousBase._setup.spacer ---
def spacer(x):
            x = x.dropna().unique()
            if len(x) < 2:
                return np.nan
            return np.min(np.diff(np.sort(x)))

# --- from mwaskom__seaborn::tests/_core/test_plot.py::TestPlotting.test_with_pyplot ---
def test_with_pyplot(self):

        p = Plot().plot(pyplot=True)

        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()
        assert p._figure is fig

# --- from okfn-brasil__serenata-de-amor::rosie/rosie/chamber_of_deputies/tests/test_traveled_speeds_classifier.py::TestTraveledSpeedsClassifier.test_predict_considers_meal_reimbursements_in_days_with_more_than_8_outliers ---
def test_predict_considers_meal_reimbursements_in_days_with_more_than_8_outliers(self):
        prediction = self.subject.predict(self.dataset)
        assert_array_equal(np.repeat(-1, 9), prediction[:9])

# --- from okfn-brasil__serenata-de-amor::rosie/rosie/chamber_of_deputies/tests/test_traveled_speeds_classifier.py::TestTraveledSpeedsClassifier.test_predict_uses_learned_thresholds_from_fit_dataset ---
def test_predict_uses_learned_thresholds_from_fit_dataset(self):
        subject = TraveledSpeedsClassifier(contamination=.6)
        subject.fit(self.dataset)
        assert_array_equal(
            np.repeat(-1, 6), subject.predict(self.dataset[13:19]))

# --- from graspologic-org__graspologic::tests/test_casc.py::TestGenCovariates.test_gen_covariates_determined ---
def test_gen_covariates_determined(self, m1=1.0, m2=0.0):
        # basic test on an identity matrix with 100% probabilities
        labels = np.array([0, 1, 2])
        X = gen_covariates(m1, m2, labels=labels, type="normal")
        assert np.array_equal(X, np.eye(3))

# --- from graspologic-org__graspologic::tests/test_casc.py::TestGenCovariates.test_gen_covariates_determined_repeated ---
def test_gen_covariates_determined_repeated(self, m1=1.0, m2=0.0):
        # test on a repeated identity matrix with 100% probabilities
        labels = np.repeat(np.array([0, 1, 2]), repeats=3)
        I = np.repeat(np.eye(3), repeats=3, axis=0)
        X = gen_covariates(m1, m2, labels=labels, type="normal")
        assert np.array_equal(X, I)

# --- from CamDavidsonPilon__lifelines::lifelines/tests/test_statistics.py::test_pairwise_allows_dataframes_and_gives_correct_counts ---
def test_pairwise_allows_dataframes_and_gives_correct_counts():
    N = 100
    N_groups = 5
    df = pd.DataFrame(np.empty((N, 3)), columns=["T", "C", "group"])
    df["T"] = np.random.exponential(1, size=N)
    df["C"] = np.random.binomial(1, 0.6, size=N)
    df["group"] = np.tile(np.arange(N_groups), 20)
    R = stats.pairwise_logrank_test(df["T"], df["group"], event_observed=df["C"])
    assert R.summary.shape[0] == N_groups * (N_groups - 1) / 2

# --- from NannyML__nannyml::nannyml/drift/univariate/methods.py::ContinuousHellingerDistance._fit ---
def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        len_reference = len(reference_data)

        bins = np.histogram_bin_edges(reference_data.astype("float64"), bins='doane')
        reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
        self._bins = bins
        self._reference_proba_in_bins = reference_proba_in_bins

        return self

# --- from tflearn__tflearn::docs/autodoc.py::get_method_doc ---
def get_method_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_method_doc(inspect.getdoc(func), header)

    if docstring != '':
        doc_source += '\n\n <span class="hr_large"></span> \n\n'
        doc_source += docstring

    return doc_source

# --- from edtechre__pybroker::src/pybroker/eval.py::r_squared ---
def r_squared(values: NDArray[np.float64]) -> float:
    """Computes R-squared of ``values``."""
    n = len(values)
    if not n:
        return 0
    x = np.arange(n)
    try:
        coeffs = np.polyfit(x, values, 1)
        pred = np.poly1d(coeffs)(x)
        y_hat = np.mean(values)
        ssres = float(np.sum((values - pred) ** 2))
        sstot = float(np.sum((values - y_hat) ** 2))
        if sstot == 0:
            return 0
        return 1 - ssres / sstot
    except Exception:
        return 0
