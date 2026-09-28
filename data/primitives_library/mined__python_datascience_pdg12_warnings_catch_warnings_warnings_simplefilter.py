# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg12::warnings.catch_warnings+warnings.simplefilter
# name: warnings_primitive
# summary: Uses warnings.catch_warnings, warnings.simplefilter across 33 repos
# anchor_symbols: ['warnings.catch_warnings', 'warnings.simplefilter']
# observed in 33 repos: ['BiomedSciAI__causallib', 'CamDavidsonPilon__lifelines', 'Data-Centric-AI-Community__fg-data-profiling', 'HazyResearch__meerkat', 'ModelOriented__DALEX']...

# --- from xorbitsai__xorbits::python/xorbits/_mars/utils.py::ignore_warning.inner ---
def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

# --- from deepchecks__deepchecks::deepchecks/core/serialization/html_display.py::HtmlDisplayableResult.widget_serializer._WidgetSerializer.serialize ---
def serialize(self, **kwargs) -> Widget:  # pylint: disable=unused-argument
                return normalize_widget_style(VBox(children=[HTML(self.value)]))

# --- from vaexio__vaex::packages/vaex-core/vaex/image.py::rgba_2_pil ---
def rgba_2_pil(rgba):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = PIL.Image.fromarray(rgba[::-1], "RGBA")  # , "RGBA", 0, -1)
    return im

# --- from alteryx__evalml::evalml/tests/component_tests/test_components.py::test_default_parameters_raise_no_warnings ---
def test_default_parameters_raise_no_warnings(cls):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cls()
        assert len(w) == 0

# --- from ploomber__sklearn-evaluation::tests/test_confusion_matrix.py::test_raw_data_doesnt_warn ---
def test_raw_data_doesnt_warn(y):
    y_true, y_pred = y

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.ConfusionMatrix.from_raw_data(y_true, y_pred)

# --- from mwaskom__seaborn::tests/test_relational.py::TestScatterPlotter.test_unfilled_marker_edgecolor_warning ---
def test_unfilled_marker_edgecolor_warning(self, long_df):  # GH2636

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scatterplot(data=long_df, x="x", y="y", marker="+")

# --- from ploomber__ploomber::tests/util/test_default.py::test_empty_reqs_mixed_envs ---
def test_empty_reqs_mixed_envs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        util.check_mixed_envs("")
        util.check_mixed_envs("nlnlyo2h3fnoun29hf2nu39ub")  # No \n in str

# --- from mljar__mljar-supervised::supervised/preprocessing/goldenfeatures_transformer.py::get_multiclass_score ---
def get_multiclass_score(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)
    ll = log_loss(y_test, pred)
    return ll

# --- from mljar__mljar-supervised::supervised/preprocessing/goldenfeatures_transformer.py::get_binary_score ---
def get_binary_score(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    ll = log_loss(y_test, pred)
    return ll

# --- from OML-Team__open-metric-learning::oml/utils/misc.py::matplotlib_backend ---
def matplotlib_backend(backend: str) -> Generator[None, None, None]:
    current_backend = matplotlib.get_backend()
    try:
        matplotlib.use(backend)
        yield
    finally:
        matplotlib.use(current_backend)

# --- from xorbitsai__xorbits::python/xorbits/_mars/utils.py::ignore_warning ---
def ignore_warning(func: Callable):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner

# --- from lgienapp__aquarel::aquarel/theme.py::Theme.__exit__ ---
def __exit__(self, exc_type, exc_val, exc_tb):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", mpl.MatplotlibDeprecationWarning)
            mpl.rcParams.update(self.rcparams_orig)
        self.apply_transforms()
