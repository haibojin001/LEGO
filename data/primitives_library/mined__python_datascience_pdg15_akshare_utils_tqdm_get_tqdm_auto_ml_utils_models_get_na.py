# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg15::akshare.utils.tqdm.get_tqdm+auto_ml.utils_models.get_name_from_model+builtins.range
# name: akshare_auto_ml_primitive
# summary: Uses akshare.utils.tqdm.get_tqdm, auto_ml.utils_models.get_name_from_model, builtins.range, causallib.contrib.adversarial_balancing.AdversarialBalancing across 19 repos
# anchor_symbols: ['akshare.utils.tqdm.get_tqdm', 'auto_ml.utils_models.get_name_from_model', 'builtins.range', 'causallib.contrib.adversarial_balancing.AdversarialBalancing', 'causallib.contrib.bicause_tree.BICauseTree', 'causallib.contrib.bicause_tree.PropensityBICauseTree']
# observed in 19 repos: ['ACEnglish__truvari', 'BiomedSciAI__causallib', 'CamDavidsonPilon__lifelines', 'ClimbsRocks__auto_ml', 'OpenDCAI__DataFlex']...

# --- from BiomedSciAI__causallib::causallib/tests/test_rlearner.py::TestRlearner.create_complex_data_for_ate_victor.m ---
def m(x, nu=0.0, gamma=1.0):
            return 0.5 / np.pi * (np.sinh(gamma)) / (
                    np.cosh(gamma) - np.cos(x - nu))

# --- from CamDavidsonPilon__lifelines::lifelines/tests/test_estimation.py::TestAalenJohansenFitter.test_jitter ---
def test_jitter(self, fitter):
        d = pd.Series([1, 1, 1])
        e = fitter._jitter(durations=d, event=pd.Series([1, 1, 1]), jitter_level=0.01)

        npt.assert_equal(np.any(np.not_equal(d, e)), True)

# --- from BiomedSciAI__causallib::causallib/contrib/hemm/hemm.py::HEMM.bernoulli_pdf ---
def bernoulli_pdf(x, mu):
        loss = nn.BCELoss(reduction='none')

        mu = mu.unsqueeze(0)
        mu = torch.clamp(mu, min=1e-3, max=1 - 1e-3)

        bern_ = -loss(mu.expand(x.shape), x)

        return torch.sum(bern_, dim=1)

# --- from deepchecks__deepchecks::tests/tabular/checks/model_evaluation/model_info_test.py::test_model_info_pipeline ---
def test_model_info_pipeline(iris_adaboost):
    # Arrange
    simple_pipeline = Pipeline([('nan_handling', SimpleImputer(strategy='most_frequent')),
                                ('adaboost', iris_adaboost)])
    # Act
    result = ModelInfo().run(simple_pipeline)
    # Assert
    assert_model_result(result)

# --- from feature-engine__feature_engine::tests/test_base_transformers/test_get_feature_names_out_mixin.py::test_with_pipe_and_skl_transformer_input_df ---
def test_with_pipe_and_skl_transformer_input_df(df_vartypes, input_features):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant")),
            ("transformer", MockTransformer()),
        ]
    )
    pipe.fit(df_vartypes)
    assert pipe.get_feature_names_out(input_features=input_features) == variables_str

# --- from feature-engine__feature_engine::tests/test_selection/test_single_feature_performance.py::test_raises_warning_if_no_feature_selected ---
def test_raises_warning_if_no_feature_selected(load_diabetes_dataset):
    X, y = load_diabetes_dataset
    sel = SelectBySingleFeaturePerformance(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=10,
    )
    with pytest.warns(UserWarning):
        sel.fit(X, y)

# --- from ploomber__sklearn-evaluation::tests/conftest.py::grid_search_2_params ---
def grid_search_2_params():
    parameters = {
        "n_estimators": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    clf.fit(X, y)

    return clf

# --- from ploomber__sklearn-evaluation::tests/conftest.py::grid_search_param_with_none ---
def grid_search_param_with_none():
    parameters = {
        "max_depth": [2, None],
        "criterion": ["gini", "entropy"],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    clf.fit(X, y)

    return clf

# --- from OpenDCAI__DataFlex::src/dataflex/offline_selector/offline_near_selector.py::FaissIndexIVFFlat.build ---
def build(self, data: np.ndarray, nprobe: int):
        data = np.ascontiguousarray(data.astype(np.float32))
        N, D = data.shape
        nlist = max(1, int(np.sqrt(N)) // 2)
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist)
        index.train(data)
        index.add(data)
        index.nprobe = nprobe
        self.index = index

# --- from OpenDCAI__DataFlex::src/dataflex/offline_selector/offline_tsds_selector.py::FaissIndexIVFFlat.build ---
def build(self, data: np.ndarray, nprobe: int):
        data = np.ascontiguousarray(data.astype(np.float32))
        N, D = data.shape
        nlist = max(1, int(np.sqrt(N)) // 2)
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist)
        index.train(data)
        index.add(data)
        index.nprobe = nprobe
        self.index = index

# --- from allenai__allennlp::tests/nn/beam_search_test.py::get_step_function._step_function ---
def _step_function(
        last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_probs_list = []
        for last_token in last_predictions:
            log_probs = torch.log(transition_matrix[last_token.item()])
            log_probs_list.append(log_probs)

        return torch.stack(log_probs_list), state

# --- from mwaskom__seaborn::tests/test_matrix.py::TestClustermap.test_square_warning ---
def test_square_warning(self):

        kws = self.default_kws.copy()
        g1 = mat.clustermap(self.df_norm, **kws)

        with pytest.warns(UserWarning):
            kws["square"] = True
            g2 = mat.clustermap(self.df_norm, **kws)

        g1_shape = g1.ax_heatmap.get_position().get_points()
        g2_shape = g2.ax_heatmap.get_position().get_points()
        assert np.array_equal(g1_shape, g2_shape)
