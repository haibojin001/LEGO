# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg5::numpy.array+numpy.asarray+numpy.testing.assert_almost_equal
# name: numpy_spacy_primitive
# summary: Uses numpy.array, numpy.asarray, numpy.testing.assert_almost_equal, numpy.testing.assert_array_equal across 2 repos
# anchor_symbols: ['numpy.array', 'numpy.asarray', 'numpy.testing.assert_almost_equal', 'numpy.testing.assert_array_equal', 'spacy.language.Language', 'spacy.util.fix_random_seed']
# observed in 2 repos: ['explosion__spaCy', 'wilsonrljr__sysidentpy']...

# --- from wilsonrljr__sysidentpy::sysidentpy/tests/test_narmax_base.py::test_polynomial_narmax_predict_fast_matches_reference_with_interactions ---
def test_polynomial_narmax_predict_fast_matches_reference_with_interactions():
    model = PredictableMSS(model_type="NARMAX")
    model.max_lag = 2
    model.n_inputs = 2
    model.basis_function = Polynomial(degree=2)
    model.final_model = np.array(
        [
            [0, 0],
            [1001, 1001],
            [2002, 1001],
            [3001, 1002],
            [3002, 2001],
        ]
    )
    model.theta = np.array([[0.3], [-0.1], [0.4], [-0.2], [0.15]])
    x_data = np.array(
        [
            [1.0, 0.5],
            [2.0, 1.5],
            [3.0, 2.5],
            [4.0, 3.5],
            [5.0, 4.5],
            [6.0, 5.5],
        ]
    )
    y_initial = np.array([[0.2], [0.4], [0.6], [0.8], [1.0], [1.2]])

    reference = model._narmax_predict_reference(
        x_data,
        y_initial,
        forecast_horizon=x_data.shape[0],
    )
    fast = model._polynomial_narmax_predict_fast(
        x_data,
        y_initial,
        forecast_horizon=x_data.shape[0],
    )

    assert_array_equal(fast.shape, reference.shape)
    assert_almost_equal(fast, reference, decimal=12)

# --- from explosion__spaCy::spacy/tests/pipeline/test_spancat.py::test_make_spangroup_singlelabel ---
def test_make_spangroup_singlelabel(threshold, allow_overlap, nr_results):
    fix_random_seed(0)
    nlp = Language()
    spancat = nlp.add_pipe(
        "spancat",
        config={
            "spans_key": SPAN_KEY,
            "threshold": threshold,
            "max_positive": 1,
        },
    )
    doc = nlp.make_doc("Greater London")
    ngram_suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes=[1, 2])
    indices = ngram_suggester([doc])[0].dataXd
    assert_array_equal(OPS.to_numpy(indices), numpy.asarray([[0, 1], [1, 2], [0, 2]]))
    labels = ["Thing", "City", "Person", "GreatCity"]
    for label in labels:
        spancat.add_label(label)
    scores = numpy.asarray(
        [[0.2, 0.4, 0.3, 0.1], [0.1, 0.6, 0.2, 0.4], [0.8, 0.7, 0.3, 0.9]], dtype="f"
    )
    spangroup = spancat._make_span_group_singlelabel(
        doc, indices, scores, allow_overlap
    )
    if threshold > 0.4:
        if allow_overlap:
            assert spangroup[0].text == "London"
            assert spangroup[0].label_ == "City"
            assert_almost_equal(0.6, spangroup.attrs["scores"][0], 5)
            assert spangroup[1].text == "Greater London"
            assert spangroup[1].label_ == "GreatCity"
            assert spangroup.attrs["scores"][1] == 0.9
            assert_almost_equal(0.9, spangroup.attrs["scores"][1], 5)
        else:
            assert spangroup[0].text == "Greater London"
            assert spangroup[0].label_ == "GreatCity"
            assert spangroup.attrs["scores"][0] == 0.9
    else:
        if allow_overlap:
            assert spangroup[0].text == "Greater"
            assert spangroup[0].label_ == "City"
            assert spangroup[1].text == "London"
            assert spangroup[1].label_ == "City"
            assert spangroup[2].text == "Greater London"
            assert spangroup[2].label_ == "GreatCity"
        else:
            assert spangroup[0].text == "Greater London"

# --- from explosion__spaCy::spacy/tests/pipeline/test_spancat.py::test_make_spangroup_multilabel ---
def test_make_spangroup_multilabel(max_positive, nr_results):
    fix_random_seed(0)
    nlp = Language()
    spancat = nlp.add_pipe(
        "spancat",
        config={"spans_key": SPAN_KEY, "threshold": 0.5, "max_positive": max_positive},
    )
    doc = nlp.make_doc("Greater London")
    ngram_suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes=[1, 2])
    indices = ngram_suggester([doc])[0].dataXd
    assert_array_equal(OPS.to_numpy(indices), numpy.asarray([[0, 1], [1, 2], [0, 2]]))
    labels = ["Thing", "City", "Person", "GreatCity"]
    for label in labels:
        spancat.add_label(label)
    scores = numpy.asarray(
        [[0.2, 0.4, 0.3, 0.1], [0.1, 0.6, 0.2, 0.4], [0.8, 0.7, 0.3, 0.9]], dtype="f"
    )
    spangroup = spancat._make_span_group_multilabel(doc, indices, scores)
    assert len(spangroup) == nr_results

    # first span is always the second token "London"
    assert spangroup[0].text == "London"
    assert spangroup[0].label_ == "City"
    assert_almost_equal(0.6, spangroup.attrs["scores"][0], 5)

    # second span depends on the number of positives that were allowed
    assert spangroup[1].text == "Greater London"
    if max_positive == 1:
        assert spangroup[1].label_ == "GreatCity"
        assert_almost_equal(0.9, spangroup.attrs["scores"][1], 5)
    else:
        assert spangroup[1].label_ == "Thing"
        assert_almost_equal(0.8, spangroup.attrs["scores"][1], 5)

    if nr_results > 2:
        assert spangroup[2].text == "Greater London"
        if max_positive == 2:
            assert spangroup[2].label_ == "GreatCity"
            assert_almost_equal(0.9, spangroup.attrs["scores"][2], 5)
        else:
            assert spangroup[2].label_ == "City"
            assert_almost_equal(0.7, spangroup.attrs["scores"][2], 5)

    assert spangroup[-1].text == "Greater London"
    assert spangroup[-1].label_ == "GreatCity"
    assert_almost_equal(0.9, spangroup.attrs["scores"][-1], 5)
