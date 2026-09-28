# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg689::sklearn.pipeline.Pipeline+sklearn.svm.SVC+sklearn_porter.Estimator.Estimator
# name: sklearn_sklearn_porter_primitive
# summary: Uses sklearn.pipeline.Pipeline, sklearn.svm.SVC, sklearn_porter.Estimator.Estimator across 2 repos
# anchor_symbols: ['sklearn.pipeline.Pipeline', 'sklearn.svm.SVC', 'sklearn_porter.Estimator.Estimator']
# observed in 2 repos: ['iterative__mlem', 'nok__sklearn-porter']...

# --- from iterative__mlem::tests/contrib/test_sklearn.py::pipeline ---
def pipeline(inp_data, out_data):
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    pipe.fit(inp_data, out_data)
    return pipe

# --- from nok__sklearn-porter::tests/EstimatorTest.py::test_unfitted_estimator_in_pipeline ---
def test_unfitted_estimator_in_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([('SVM', SVC())])
    with pytest.raises(exception.NotFittedEstimatorError):
        Estimator(pipeline)

# --- from nok__sklearn-porter::tests/EstimatorTest.py::test_estimator_extraction_from_pipeline ---
def test_estimator_extraction_from_pipeline():
    """Test the extraction of an estimator from a pipeline."""
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([('SVM', SVC())])
    pipeline.fit(X=[[1, 1], [1, 1], [2, 2]], y=[1, 1, 2])
    est = Estimator(pipeline)
    assert isinstance(est.estimator, SVC)
