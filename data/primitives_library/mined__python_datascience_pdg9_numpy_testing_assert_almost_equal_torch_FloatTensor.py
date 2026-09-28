# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg9::numpy.testing.assert_almost_equal+torch.FloatTensor
# name: numpy_torch_primitive
# summary: Uses numpy.testing.assert_almost_equal, torch.FloatTensor across 3 repos
# anchor_symbols: ['numpy.testing.assert_almost_equal', 'torch.FloatTensor']
# observed in 3 repos: ['allenai__allennlp', 'explosion__spaCy', 'xorbitsai__xorbits']...

# --- from allenai__allennlp::tests/modules/attention/cosine_attention_test.py::TestCosineAttention.test_cosine_similarity ---
def test_cosine_similarity(self):
        linear = CosineAttention(normalize=False)
        output = linear(
            torch.FloatTensor([[0, 0, 0], [1, 1, 1]]),
            torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        )

        assert_almost_equal(output.numpy(), numpy.array([[0.0, 0.0], [0.9948, 0.9973]]), decimal=2)

# --- from allenai__allennlp::tests/modules/attention/dot_product_attention_test.py::TestDotProductAttention.test_dot_product_similarity ---
def test_dot_product_similarity(self):
        attn = DotProductAttention(normalize=False)
        output = attn(
            torch.FloatTensor([[0, 0, 0], [1, 1, 1]]),
            torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        )

        assert_almost_equal(output.numpy(), numpy.array([[0.0, 0.0], [24.0, 33.0]]), decimal=2)

# --- from xorbitsai__xorbits::python/xorbits/_mars/learn/metrics/tests/test_classification.py::test_precision_recall_f1_no_labels_check_warnings ---
def test_precision_recall_f1_no_labels_check_warnings(setup, average):
    y_true = mt.zeros((20, 3))
    y_pred = mt.zeros_like(y_true)

    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)
        p, r, f = fetch(execute(p, r, f))

    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fetch(execute(fbeta_score(y_true, y_pred, average=average, beta=1.0)))

    assert_almost_equal(fbeta, 0)

# --- from xorbitsai__xorbits::python/xorbits/_mars/learn/linear_model/tests/test_base.py::test_linear_regression_sample_weights ---
def test_linear_regression_sample_weights(setup):
    # TODO: loop over sparse data as well

    rng = np.random.RandomState(0)

    # It would not work with under-determined systems
    for n_samples, n_features in ((6, 5),):
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)
        sample_weight = 1.0 + rng.rand(n_samples)

        for intercept in (True, False):
            # LinearRegression with explicit sample_weight
            reg = LinearRegression(fit_intercept=intercept)
            reg.fit(X, y, sample_weight=sample_weight)
            coefs1 = reg.coef_
            inter1 = reg.intercept_

            assert reg.coef_.shape == (X.shape[1],)  # sanity checks
            assert reg.score(X, y).to_numpy() > 0.5

            # Closed form of the weighted least square
            # theta = (X^T W X)^(-1) * X^T W y
            W = np.diag(sample_weight)
            if intercept is False:
                X_aug = X
            else:
                dummy_column = np.ones(shape=(n_samples, 1))
                X_aug = np.concatenate((dummy_column, X), axis=1)

            coefs2 = linalg.solve(X_aug.T.dot(W).dot(X_aug), X_aug.T.dot(W).dot(y))

            if intercept is False:
                assert_array_almost_equal(coefs1, coefs2)
            else:
                assert_array_almost_equal(coefs1, coefs2[1:])
                assert_almost_equal(inter1.to_numpy(), coefs2[0])

# --- from explosion__spaCy::spacy/tests/test_scorer.py::test_roc_auc_score ---
def test_roc_auc_score():
    # Binary classification, toy tests from scikit-learn test suite
    y_true = [0, 1]
    y_score = [0, 1]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    # same result as above with ROCAUCScore wrapper
    score = ROCAUCScore()
    score.score_set(0.5, 1)
    score.score_set(0.5, 0)
    assert_almost_equal(score.score, 0.5)

    # check that errors are raised in undefined cases and score is -inf
    y_true = [0, 0]
    y_score = [0.25, 0.75]
    with pytest.raises(ValueError):
        _roc_auc_score(y_true, y_score)

    score = ROCAUCScore()
    score.score_set(0.25, 0)
    score.score_set(0.75, 0)
    with pytest.raises(ValueError):
        _ = score.score  # noqa: F841

    y_true = [1, 1]
    y_score = [0.25, 0.75]
    with pytest.raises(ValueError):
        _roc_auc_score(y_true, y_score)

    score = ROCAUCScore()
    score.score_set(0.25, 1)
    score.score_set(0.75, 1)
    with pytest.raises(ValueError):
        _ = score.score  # noqa: F841

# --- from explosion__spaCy::spacy/tests/vocab_vectors/test_vectors.py::test_floret_vectors ---
def test_floret_vectors(floret_vectors_vec_str, floret_vectors_hashvec_str):
    nlp = English()
    nlp_plain = English()
    # load both vec and hashvec tables
    with make_tempdir() as tmpdir:
        p = tmpdir / "test.hashvec"
        with open(p, "w") as fileh:
            fileh.write(floret_vectors_hashvec_str)
        convert_vectors(nlp, p, truncate=0, prune=-1, mode="floret")
        p = tmpdir / "test.vec"
        with open(p, "w") as fileh:
            fileh.write(floret_vectors_vec_str)
        convert_vectors(nlp_plain, p, truncate=0, prune=-1)

    word = "der"
    # ngrams: full padded word + padded 2-grams + padded 3-grams
    ngrams = nlp.vocab.vectors._get_ngrams(word)
    assert ngrams == ["<der>", "<d", "de", "er", "r>", "<de", "der", "er>"]
    # rows: 2 rows per ngram
    rows = OPS.xp.asarray(
        [
            h % nlp.vocab.vectors.shape[0]
            for ngram in ngrams
            for h in nlp.vocab.vectors._get_ngram_hashes(ngram)
        ],
        dtype="uint32",
    )
    assert_equal(
        OPS.to_numpy(rows),
        numpy.asarray([5, 6, 7, 5, 8, 2, 8, 9, 3, 3, 4, 6, 7, 3, 0, 2]),
    )
    assert len(rows) == len(ngrams) * nlp.vocab.vectors.hash_count
    # all vectors are equivalent for plain static table vs. hash ngrams
    for word in nlp_plain.vocab.vectors:
        word = nlp_plain.vocab.strings.as_string(word)
        assert_almost_equal(
            nlp.vocab[word].vector, nlp_plain.vocab[word].vector, decimal=3
        )

        # every word has a vector
        assert nlp.vocab[word * 5].has_vector

    # n_keys is -1 for floret
    assert nlp_plain.vocab.vectors.n_keys > 0
    assert nlp.vocab.vectors.n_keys == -1

    # check that single and batched vector lookups are identical
    words = [s for s in nlp_plain.vocab.vectors]
    single_vecs = OPS.to_numpy(OPS.asarray([nlp.vocab[word].vector for word in words]))
    batch_vecs = OPS.to_numpy(nlp.vocab.vectors.get_batch(words))
    assert_equal(single_vecs, batch_vecs)

    # an empty key returns 0s
    assert_equal(
        OPS.to_numpy(nlp.vocab[""].vector),
        numpy.zeros((nlp.vocab.vectors.shape[0],)),
    )
    # an empty batch returns 0s
    assert_equal(
        OPS.to_numpy(nlp.vocab.vectors.get_batch([""])),
        numpy.zeros((1, nlp.vocab.vectors.shape[0])),
    )
    # an empty key within a batch returns 0s
    assert_equal(
        OPS.to_numpy(nlp.vocab.vectors.get_batch(["a", "", "b"])[1]),
        numpy.zeros((nlp.vocab.vectors.shape[0],)),
    )

    # the loaded ngram vector table cannot be modified
    # except for clear: warning, then return without modifications
    vector = list(range(nlp.vocab.vectors.shape[1]))
    orig_bytes = nlp.vocab.vectors.to_bytes(exclude=["strings"])
    with pytest.warns(UserWarning):
        nlp.vocab.set_vector("the", vector)
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=["strings"])
    with pytest.warns(UserWarning):
        nlp.vocab[word].vector = vector
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=["strings"])
    with pytest.warns(UserWarning):
        nlp.vocab.vectors.add("the", row=6)
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=["strings"])
    with pytest.warns(UserWarning):
        nlp.vocab.vectors.resize(shape=(100, 10))
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=["strings"])
    with pytest.raises(ValueError):
        nlp.vocab.vectors.clear()

    # data and settings are serialized correctly
    with make_tempdir() as d:
        nlp.vocab.to_disk(d)
        vocab_r = Vocab()
        vocab_r.from_disk(d)
        assert nlp.vocab.vectors.to_bytes() == vocab_r.vectors.to_bytes()
        assert_equal(
            OPS.to_numpy(nlp.vocab.vectors.data), OPS.to_numpy(vocab_r.vectors.data)
        )
        assert_equal(nlp.vocab.vectors._get_cfg(), vocab_r.vectors._get_cfg())
        assert_almost_equal(
            OPS.to_numpy(nlp.vocab[word].vector),
            OPS.to_numpy(vocab_r[word].vector),
            decimal=6,
        )
