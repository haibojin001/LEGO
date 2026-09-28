# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg298::category_encoders.CountEncoder+collections.OrderedDict+copy.copy
# name: category_encoders_collections_primitive
# summary: Uses category_encoders.CountEncoder, collections.OrderedDict, copy.copy, copy.deepcopy across 13 repos
# anchor_symbols: ['category_encoders.CountEncoder', 'collections.OrderedDict', 'copy.copy', 'copy.deepcopy', 'dalex.Aspect', 'dalex.Explainer']
# observed in 13 repos: ['BiomedSciAI__causallib', 'HazyResearch__meerkat', 'ModelOriented__DALEX', 'NannyML__nannyml', 'ahmetozlu__tensorflow_object_counting_api']...

# --- from lazyprogrammer__machine_learning_examples::unsupervised_class3/bayes_classifier_gaussian.py::clamp_sample ---
def clamp_sample(x):
  x = np.minimum(x, 1)
  x = np.maximum(x, 0)
  return x

# --- from lazyprogrammer__machine_learning_examples::unsupervised_class3/bayes_classifier_gmm.py::clamp_sample ---
def clamp_sample(x):
  x = np.minimum(x, 1)
  x = np.maximum(x, 0)
  return x

# --- from ahmetozlu__tensorflow_object_counting_api::utils/object_tracking_module/tracking_utils.py::convert_to_cv2bbox ---
def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])
    
    return (left, top, right, bottom)

# --- from BiomedSciAI__causallib::causallib/evaluation/plots/plots.py::_calculate_mutual_bins ---
def _calculate_mutual_bins(x, y, bins="auto"):
    """
    A common support for two vectors.

    Args:
        x (pd.Series):
        y (pd.Series):
        bins: compatible with numpy's bins parameter.

    Returns:
        np.array: bins cutoffs.
    """
    data = np.append(x, y)
    bins = np.histogram(data, bins=bins)[1]
    return bins

# --- from ahmetozlu__tensorflow_object_counting_api::utils/object_tracking_module/tracking_utils.py::box_iou2 ---
def box_iou2(a, b):    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)

# --- from ModelOriented__DALEX::python/dalex/test/test_shap_wrapper.py::TestShapWrapperRandomForestClassifierTitanicNumericalDataset.setUp ---
def setUp(self):
        data = dx.datasets.load_titanic()
        data.loc[:, 'survived'] = LabelEncoder().fit_transform(data.survived)

        self.X = data.loc[:, ["age", "fare", "sibsp", "parch"]]
        self.y = data.survived

        clf = RandomForestClassifier(n_estimators=100, random_state=123)
        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)

# --- from zama-ai__concrete-ml::src/concrete/ml/onnx/ops_impl.py::numpy_hardswish ---
def numpy_hardswish(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute hardswish in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#hardswish-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    alpha = 1.0 / 6
    beta = 0.5
    r = x * numpy.maximum(0, numpy.minimum(1, alpha * x + beta))

    return (r,)

# --- from zama-ai__concrete-ml::src/concrete/ml/onnx/ops_impl.py::numpy_celu ---
def numpy_celu(x: numpy.ndarray, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
    """Compute celu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Celu-12

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.maximum(0, x) + numpy.minimum(0, alpha * (numpy.exp(x / alpha) - 1)),)

# --- from NannyML__nannyml::nannyml/drift/univariate/methods.py::ContinuousJensenShannonDistance._calculate ---
def _calculate(self, data: pd.Series):
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        data = _remove_nans(data)
        if data.empty:
            return np.nan

        len_data = len(data)
        data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len_data

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = jensenshannon(reference_proba_in_bins, data_proba_in_bins, base=2)

        return distance

# --- from NannyML__nannyml::nannyml/drift/univariate/methods.py::ContinuousHellingerDistance._calculate ---
def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = np.sqrt(np.sum((np.sqrt(reference_proba_in_bins) - np.sqrt(data_proba_in_bins)) ** 2)) / np.sqrt(2)

        return distance

# --- from mljar__mljar-supervised::supervised/utils/shap.py::PlotSHAP.get_explainer ---
def get_explainer(algorithm, X_train):
        explainer = None
        if algorithm.algorithm_short_name in [
            "Xgboost",
            "Decision Tree",
            "Random Forest",
            "LightGBM",
            "Extra Trees",
            "CatBoost",
        ]:
            explainer = shap.TreeExplainer(algorithm.model)
        elif algorithm.algorithm_short_name in ["Linear"]:
            explainer = shap.LinearExplainer(algorithm.model, X_train)
        # elif algorithm.algorithm_short_name in ["Neural Network"]:
        #    explainer = shap.KernelExplainer(algorithm.model.predict, X_train)  # slow

        return explainer

# --- from feature-engine__feature_engine::tests/test_pipeline/test_pipeline_sklearn.py::test_fit_predict_on_pipeline_without_fit_predict ---
def test_fit_predict_on_pipeline_without_fit_predict():
    # tests that a pipeline does not have fit_predict method when final
    # step of pipeline does not have fit_predict defined
    scaler = StandardScaler()
    pca = PCA(svd_solver="full")
    pipe = Pipeline([("scaler", scaler), ("pca", pca)])

    outer_msg = "'Pipeline' has no attribute 'fit_predict'"
    inner_msg = "'PCA' object has no attribute 'fit_predict'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        getattr(pipe, "fit_predict")
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)
