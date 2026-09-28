# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg246::deepchecks.tabular.checks.SingleDatasetPerformance+deepchecks.tabular.checks.model_evaluation.TrainTestPerformance+functools.reduce
# name: deepchecks_functools_primitive
# summary: Uses deepchecks.tabular.checks.SingleDatasetPerformance, deepchecks.tabular.checks.model_evaluation.TrainTestPerformance, functools.reduce, gluonts.dataset.util.forecast_start across 9 repos
# anchor_symbols: ['deepchecks.tabular.checks.SingleDatasetPerformance', 'deepchecks.tabular.checks.model_evaluation.TrainTestPerformance', 'functools.reduce', 'gluonts.dataset.util.forecast_start', 'gluonts.model.forecast.SampleForecast', 'hamcrest.assert_that']
# observed in 9 repos: ['NorskRegnesentral__skweak', 'awslabs__gluonts', 'cleanlab__cleanlab', 'deepchecks__deepchecks', 'iterative__mlem']...

# --- from iterative__mlem::tests/contrib/test_pandas.py::iris_data ---
def iris_data():
    data, y = load_iris(return_X_y=True, as_frame=True)
    data["target"] = y
    train_data, _ = train_test_split(data, random_state=42)
    return train_data

# --- from rasbt__mlxtend::mlxtend/feature_selection/tests/test_sequential_feature_selector.py::test_run_default ---
def test_run_default():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    sfs = SFS(estimator=knn, verbose=0)
    sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (3,)

# --- from rasbt__mlxtend::mlxtend/feature_selection/tests/test_sequential_feature_selector_feature_groups.py::test_run_default ---
def test_run_default():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    sfs = SFS(estimator=knn, verbose=0)
    sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (3,)

# --- from reiinakano__scikit-plot::scikitplot/tests/test_classifiers.py::TestPlotKSStatistic.test_two_classes ---
def test_two_classes(self):
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        X, y = load_data(return_X_y=True)
        self.assertRaises(ValueError, clf.plot_ks_statistic, X, y)

# --- from cleanlab__cleanlab::cleanlab/datalab/internal/spurious_correlation.py::_train_and_eval ---
def _train_and_eval(X, y, cv=5) -> float:
    classifier = GaussianNB()  # TODO: Make this a parameter
    cv_accuracies = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")
    mean_accuracy = float(np.mean(cv_accuracies))
    return mean_accuracy

# --- from reiinakano__scikit-plot::scikitplot/tests/test_metrics.py::TestPlotLiftCurve.test_two_classes ---
def test_two_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        self.assertRaises(ValueError, plot_lift_curve, y, probas)

# --- from ploomber__sklearn-evaluation::tests/test_cumulative_gain_lift_curve.py::test_two_classes_lift_curve ---
def test_two_classes_lift_curve(ploomber_value_error_message):
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    with pytest.raises(ValueError, match=ploomber_value_error_message) as e:
        lift_curve(y, probas)
    assert "Cannot calculate Lift Curve for data with 3 category/ies" in str(e.value)

# --- from ploomber__sklearn-evaluation::tests/test_classifier_evaluator.py::test_can_plot ---
def test_can_plot():
    data = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)
    X = data[0]
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    est = RandomForestClassifier()
    est.fit(X_train, y_train)

    evaluator = ClassifierEvaluator(est, y_true=y_test, X=X_test)

    evaluator.confusion_matrix()

# --- from NorskRegnesentral__skweak::skweak/aggregation.py::AbstractAggregator.filter_observations ---
def filter_observations(self, obs:pandas.DataFrame) -> pandas.DataFrame:
        
        # We count the votes for each label on all sources 
        def count_fun(x):
            return np.bincount(x[x>=0], minlength=len(self.observed_labels)) 
        
        label_votes = np.apply_along_axis(count_fun, 1, obs.values).astype(np.float32)
        out_label_votes = label_votes.dot(self._get_vote_matrix())
        relevant_rows = (out_label_votes.sum(axis=1) > 0.0)
        
        return obs[relevant_rows] #type: ignore

# --- from iterative__mlem::tests/contrib/test_sklearn.py::test_preprocess_transformer ---
def test_preprocess_transformer(
    classifier, transformer_fixture, inp_data, tmpdir, out_data, request
):
    transformer = request.getfixturevalue(transformer_fixture)
    model_file = "clf"
    clf = LogisticRegression()
    train_data = transformer.transform(inp_data)
    clf.fit(train_data, out_data)
    save(
        clf,
        str(tmpdir / model_file),
        sample_data=inp_data,
        preprocess=transformer,
    )
    clf = load_meta(str(tmpdir / model_file))
    output = apply(clf, inp_data)
    assert np.array_equal(output, out_data)

# --- from deepchecks__deepchecks::tests/tabular/checks/model_evaluation/train_test_performance_test.py::test_classification_binary ---
def test_classification_binary(iris_dataset_single_class_labeled):
    # Arrange
    train, test = train_test_split(iris_dataset_single_class_labeled.data, test_size=0.33, random_state=42)
    train_ds = iris_dataset_single_class_labeled.copy(train)
    test_ds = iris_dataset_single_class_labeled.copy(test)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
    check = TrainTestPerformance()

    # Act
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification_result(result, test_ds)

# --- from jasmcaus__caer::examples/GUI/caer_gui.py::show_histogram_window ---
def show_histogram_window():
    # reset the error label's text
    if lblError['text'] == 'Error':
        lblError['text'] = ''

    plt.close()

    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixels')

    colors = ('r', 'g', 'b')

    for i,col in enumerate(colors):
        if not transformedImage is None:
            img = transformedImage
        else:
            img = currentImage

        hist = caer.core.cv.calcHist([img], [i], None, [256], [0,256])

        plt.plot(hist, color=col)
        plt.xlim([0,256])

    plt.show()
