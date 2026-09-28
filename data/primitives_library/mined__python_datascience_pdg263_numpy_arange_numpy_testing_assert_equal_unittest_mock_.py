# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg263::numpy.arange+numpy.testing.assert_equal+unittest.mock.MagicMock
# name: numpy_unittest_primitive
# summary: Uses numpy.arange, numpy.testing.assert_equal, unittest.mock.MagicMock across 3 repos
# anchor_symbols: ['numpy.arange', 'numpy.testing.assert_equal', 'unittest.mock.MagicMock']
# observed in 3 repos: ['modin-project__modin', 'wilsonrljr__sysidentpy', 'yzhao062__pyod']...

# --- from modin-project__modin::modin/tests/pandas/test_groupby.py::test_groupby_getitem_preserves_key_order_issue_6154 ---
def test_groupby_getitem_preserves_key_order_issue_6154():
    a = np.tile(["a", "b", "c", "d", "e"], (1, 10))
    np.random.shuffle(a[0])
    df = pd.DataFrame(
        np.hstack((a.T, np.arange(100).reshape((50, 2)))),
        columns=["col 1", "col 2", "col 3"],
    )
    eval_general(
        df, df._to_pandas(), lambda df: df.groupby("col 1")[["col 3", "col 2"]].count()
    )

# --- from wilsonrljr__sysidentpy::sysidentpy/general_estimators/tests/test_general_narx.py::test_nar_n_step_prediction_path ---
def test_nar_n_step_prediction_path():
    model = fit_narx_model(model_type="NAR", x_data=None)
    model._model_prediction = MagicMock(
        return_value=np.arange(100, dtype=float).reshape(-1, 1)
    )
    steps = 2
    yhat = model._n_step_ahead_prediction(None, y_test, steps_ahead=steps)
    expected = y_test.shape[0] + steps - model.max_lag
    assert_equal(yhat.shape[0], expected)

# --- from wilsonrljr__sysidentpy::sysidentpy/general_estimators/tests/test_general_narx.py::test_nar_step_ahead_multi_segment_prediction ---
def test_nar_step_ahead_multi_segment_prediction():
    model = fit_narx_model(model_type="NAR", x_data=None)
    model._model_prediction = MagicMock(
        return_value=np.arange(100, dtype=float).reshape(-1, 1)
    )
    steps = 3
    yhat = model._nar_step_ahead(y_test, steps_ahead=steps)
    expected = y_test.shape[0] + steps - model.max_lag
    assert_equal(yhat.shape[0], expected)
    assert_equal(model._model_prediction.called, True)

# --- from yzhao062__pyod::pyod/test/test_encoders.py::TestOpenAIEncoder.test_encode_shape ---
def test_encode_shape(self, mock_openai_cls):
        from pyod.utils.encoders.openai_encoder import OpenAIEncoder

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = \
            self._make_mock_response(3)

        encoder = OpenAIEncoder(model_name='text-embedding-3-small')
        emb = encoder.encode(["text1", "text2", "text3"])
        assert_equal(emb.shape, (3, 1536))

# --- from yzhao062__pyod::pyod/test/test_encoders.py::TestOpenAIEncoder.test_encode_batching ---
def test_encode_batching(self, mock_openai_cls):
        from pyod.utils.encoders.openai_encoder import OpenAIEncoder

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def side_effect(**kwargs):
            n = len(kwargs['input'])
            return self._make_mock_response(n)

        mock_client.embeddings.create.side_effect = side_effect

        encoder = OpenAIEncoder(model_name='text-embedding-3-small')
        texts = [f"text_{i}" for i in range(3000)]
        emb = encoder.encode(texts)
        assert_equal(emb.shape[0], 3000)
        assert_equal(mock_client.embeddings.create.call_count, 2)
