# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg170::numpy.testing.assert_array_equal
# name: numpy_primitive
# summary: Uses numpy.testing.assert_array_equal across 4 repos
# anchor_symbols: ['numpy.testing.assert_array_equal']
# observed in 4 repos: ['alegonz__baikal', 'awslabs__gluonts', 'mwaskom__seaborn', 'xorbitsai__xorbits']...

# --- from mwaskom__seaborn::seaborn/_core/scales.py::_make_power_transforms.forward ---
def forward(x):
        return np.sign(x) * np.power(np.abs(x), exp)

# --- from mwaskom__seaborn::seaborn/_core/scales.py::_make_power_transforms.inverse ---
def inverse(x):
        return np.sign(x) * np.power(np.abs(x), 1 / exp)

# --- from alegonz__baikal::tests/steps/test_merge.py::test_stack ---
def test_stack(teardown):
    x1 = Input()
    x2 = Input()
    y = Stack(axis=1)([x1, x2])
    model = Model([x1, x2], y)

    x1_data = np.array([[1, 2], [10, 20]])
    x2_data = np.array([[3, 4], [30, 40]])
    y_expected = np.stack([x1_data, x2_data], axis=1)
    y_pred = model.predict([x1_data, x2_data])

    assert_array_equal(y_pred, y_expected)

# --- from awslabs__gluonts::src/gluonts/nursery/robust-mts-attack/pts/modules/gaussian_diffusion.py::cosine_beta_schedule ---
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# --- from awslabs__gluonts::test/transform/test_add_time_features.py::compute_time_features ---
def compute_time_features(
    entry: Dict,
    time_features: List[TimeFeature],
    pred_length: int = 0,
    dtype=np.float32,
):
    assert pred_length >= 0
    index = to_pandas(entry, freq=entry["start"].freq).index

    if pred_length > 0:
        index = index.union(
            pd.period_range(
                index[-1] + 1, index[-1] + pred_length, freq=index.freq
            )
        )

    feature_arrays = [feat(index) for feat in time_features]
    return np.vstack(feature_arrays).astype(dtype)

# --- from xorbitsai__xorbits::python/xorbits/_mars/learn/metrics/tests/test_regression.py::test__check_reg_targets ---
def test__check_reg_targets(setup):
    # All of length 3
    EXAMPLES = [
        ("continuous", [1, 2, 3], 1),
        ("continuous", [[1], [2], [3]], 1),
        ("continuous-multioutput", [[1, 1], [2, 2], [3, 1]], 2),
        ("continuous-multioutput", [[5, 1], [4, 2], [3, 1]], 2),
        ("continuous-multioutput", [[1, 3, 4], [2, 2, 2], [3, 1, 1]], 3),
    ]

    for (type1, y1, n_out1), (type2, y2, n_out2) in product(EXAMPLES, repeat=2):
        if type1 == type2 and n_out1 == n_out2:
            y_type, y_check1, y_check2, multioutput = _check_reg_targets(y1, y2, None)
            assert type1 == y_type
            if type1 == "continuous":
                assert_array_equal(y_check1, np.reshape(y1, (-1, 1)))
                assert_array_equal(y_check2, np.reshape(y2, (-1, 1)))
            else:
                assert_array_equal(y_check1, y1)
                assert_array_equal(y_check2, y2)
        else:
            with pytest.raises(ValueError):
                _check_reg_targets(y1, y2, None)

# --- from alegonz__baikal::tests/_core/test_model.py::test_fit_predict_naive_stack ---
def test_fit_predict_naive_stack(teardown):
    x_data = iris.data
    y_t_data = iris.target
    random_state = 123

    # baikal way
    x = Input()
    y_t = Input()
    y1 = LogisticRegression(random_state=random_state, solver="liblinear")(x, y_t)
    y2 = RandomForestClassifier(random_state=random_state)(x, y_t)
    features = Stack(axis=1)([y1, y2])
    y = LogisticRegression(random_state=random_state, solver="liblinear")(features, y_t)

    model = Model(x, y, y_t)
    model.fit(x_data, y_t_data)
    y_pred_baikal = model.predict(x_data)

    # traditional way
    logreg = LogisticRegression(random_state=random_state, solver="liblinear")
    logreg.fit(x_data, y_t_data)
    logreg_pred = logreg.predict(x_data)

    random_forest = RandomForestClassifier(random_state=random_state)
    random_forest.fit(x_data, y_t_data)
    random_forest_pred = random_forest.predict(x_data)

    features = np.stack([logreg_pred, random_forest_pred], axis=1)
    stacked = LogisticRegression(random_state=random_state, solver="liblinear")
    stacked.fit(features, y_t_data)
    y_pred_traditional = stacked.predict(features)

    assert_array_equal(y_pred_baikal, y_pred_traditional)
