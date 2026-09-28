# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg167::pandas.DataFrame
# name: pandas_primitive
# summary: Uses pandas.DataFrame across 11 repos
# anchor_symbols: ['pandas.DataFrame']
# observed in 11 repos: ['CamDavidsonPilon__lifelines', 'CamDavidsonPilon__lifetimes', 'alteryx__evalml', 'alteryx__featuretools', 'awslabs__gluonts']...

# --- from mwaskom__seaborn::tests/test_axisgrid.py::TestPairGrid.test_ignore_datelike_data ---
def test_ignore_datelike_data(self):

        df = self.df.copy()
        df['date'] = pd.date_range('2010-01-01', periods=len(df), freq='D')
        result = ag.PairGrid(self.df).data
        expected = df.drop('date', axis=1)
        tm.assert_frame_equal(result, expected)

# --- from mwaskom__seaborn::tests/test_regression.py::TestLinearPlotter.test_establish_variables_from_frame ---
def test_establish_variables_from_frame(self):

        p = lm._LinearPlotter()
        p.establish_variables(self.df, x="x", y="y")
        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        pdt.assert_frame_equal(p.data, self.df)

# --- from feature-engine__feature_engine::tests/test_dataframe_checks.py::test_check_X_y_returns_pandas_from_pandas_with_non_typical_index ---
def test_check_X_y_returns_pandas_from_pandas_with_non_typical_index():
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_series_equal(s, y)

# --- from alteryx__featuretools::featuretools/tests/primitive_tests/test_feature_serialization.py::test_serialize_url ---
def test_serialize_url(es):
    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=es,
        features_only=True,
    )
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        save_features(features_original, URL)

# --- from alteryx__featuretools::featuretools/tests/computational_backend/test_calculate_feature_matrix.py::test_no_relationships ---
def test_no_relationships(dataframes):
    fm_1, features = dfs(
        dataframes=dataframes,
        relationships=None,
        target_dataframe_name="transactions",
    )

    fm_2 = calculate_feature_matrix(
        features=features,
        dataframes=dataframes,
        relationships=None,
    )

    assert fm_1.equals(fm_2)

# --- from alteryx__evalml::evalml/tests/component_tests/test_baseline_regressor.py::test_baseline_median ---
def test_baseline_median(X_y_regression):
    X, y = X_y_regression
    median = np.median(y)
    clf = BaselineRegressor(strategy="median")
    clf.fit(X, y)

    expected_predictions = pd.Series([median] * len(X))
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))

# --- from feature-engine__feature_engine::tests/test_dataframe_checks.py::test_check_x_y_returns_pandas_from_pandas ---
def test_check_x_y_returns_pandas_from_pandas(df_vartypes):
    # when s is series
    s = pd.Series([0, 1, 2, 3])
    x, y = check_X_y(df_vartypes, s)
    assert_frame_equal(df_vartypes, x)
    assert_series_equal(s, y)

    # when y is multioutput
    d = pd.DataFrame(np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2))
    x, y = check_X_y(df_vartypes, d)
    assert_frame_equal(df_vartypes, x)
    assert_frame_equal(d, y)

# --- from alteryx__evalml::evalml/tests/component_tests/test_target_imputer.py::test_target_imputer_with_X ---
def test_target_imputer_with_X():
    X = pd.DataFrame({"some col": [1, 3, np.nan]})
    y = pd.Series([np.nan, 1, 3])
    imputer = TargetImputer(impute_strategy="median")
    y_expected = pd.Series([2, 1, 3])
    X_expected = pd.DataFrame({"some col": [1, 3, np.nan]})
    X_t, y_t = imputer.fit_transform(X, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)
    assert_frame_equal(X_expected, X_t, check_dtype=False)

# --- from cleanlab__cleanvision::tests/test_save_load.py::TestImagelabSaveLoad.compare_dict ---
def compare_dict(self, a, b):
        assert len(a) == len(b)
        for k, v in a.items():
            print(k)
            assert k in b
            if isinstance(v, dict):
                self.compare_dict(v, b[k])
            elif isinstance(v, pd.DataFrame):
                assert_frame_equal(v, b[k])
            elif isinstance(v, pd.Series):
                assert_series_equal(v, b[k])
            else:
                assert v == b[k]

# --- from microsoft__RD-Agent::rdagent/scenarios/qlib/proposal/bandit.py::LinearThompsonTwoArm.sample_reward ---
def sample_reward(self, arm: str, x: np.ndarray) -> float:
        P = self.precision[arm]
        P = 0.5 * (P + P.T)

        eps = 1e-6
        try:
            cov = np.linalg.inv(P + eps * np.eye(self.dim))
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            w_sample = self.mean[arm]

        return float(np.dot(w_sample, x))

# --- from awslabs__gluonts::test/zebras/test_timeframe.py::test_time_frame ---
def test_time_frame():
    assert tf.eq_to(tf)
    assert len(tf) == 10
    assert len(tf[:4]) == 4
    assert len(tf[-4:]) == 4
    assert tf.tdims

    assert tf["target"].name == "target"
    assert np.array_equal(tf["target"], target)
    assert np.array_equal(tf["feat"], feat)

    assert tf.metadata == {"x": 42}

    tf2 = tf.stack(["target", "feat"], "stacked")
    assert not tf.eq_to(tf2)
    assert tf2.columns["stacked"].shape == (3, 10)
    assert tf2.columns["stacked"].shape == (3, 10)
    assert "target" not in tf2.columns
    assert "feat" not in tf2.columns
    assert tf2.metadata == {"x": 42}
    assert tf2.tdims

    tf3 = tf.like({"foo": np.full(10, 7)})
    assert tf3.columns["foo"].shape == (10,)
    assert "target" not in tf3.columns
    assert "feat" not in tf3.columns
    assert tf3.metadata == {"x": 42}
    assert tf3.tdims

# --- from CamDavidsonPilon__lifelines::lifelines/tests/test_estimation.py::TestCustomRegressionModel.test_reparameterization_flips_the_sign ---
def test_reparameterization_flips_the_sign(self, rossi):

        regressors = {"lambda_": rossi.columns.difference(["arrest", "week"]), "rho_": "1", "beta_": "fin + 1"}

        cmA = CureModelA()
        cmB = CureModelB()
        cmC = CureModelC()

        cmA.fit(rossi, "week", event_col="arrest", regressors=regressors)
        cmB.fit(rossi, "week", event_col="arrest", regressors=regressors)
        cmC.fit(
            rossi,
            "week",
            event_col="arrest",
            regressors={"lambda_": rossi.columns.difference(["week", "arrest"]), "rho_": "1", "beta_": "fin + 1"},
        )
        assert_frame_equal(cmA.summary.loc["lambda_"], cmB.summary.loc["lambda_"])
        assert_frame_equal(cmA.summary.loc["rho_"], cmB.summary.loc["rho_"])
        assert_frame_equal(cmC.summary, cmB.summary)
        assert_series_equal(cmA.params_.loc["beta_"], -cmB.params_.loc["beta_"])
