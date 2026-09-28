# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg1::pickle.dumps+pickle.loads
# name: pickle_primitive
# summary: Uses pickle.dumps, pickle.loads across 19 repos
# anchor_symbols: ['pickle.dumps', 'pickle.loads']
# observed in 19 repos: ['BiomedSciAI__causallib', 'CamDavidsonPilon__lifelines', 'Lightning-AI__torchmetrics', 'allenai__allennlp', 'alteryx__featuretools']...

# --- from mahmoud__boltons::tests/test_namedutils.py::test_namedtuple_pickle ---
def test_namedtuple_pickle():
    p = Point(x=10, y=20)
    assert p == loads(dumps(p))

# --- from mahmoud__boltons::tests/test_namedutils.py::test_namedlist_pickle ---
def test_namedlist_pickle():
    p = MutablePoint(x=10, y=20)
    assert p == loads(dumps(p))

# --- from ploomber__ploomber::tests/test_pickle.py::test_file_is_pickable ---
def test_file_is_pickable():
    f = File("/path/to/file.csv")
    pickle.loads(pickle.dumps(f))

# --- from ploomber__ploomber::tests/test_pickle.py::test_placeholder_is_picklable ---
def test_placeholder_is_picklable():
    p = Placeholder("{{hi}}")
    pickle.loads(pickle.dumps(p))

# --- from pydoit__doit::tests/test_cmdparse.py::TestDefaultUpdate.test_pickle ---
def test_pickle(self):
        du = DefaultUpdate()
        du.set_default('x', 0)
        dump = pickle.dumps(du, 2)
        pickle.loads(dump)

# --- from allenai__allennlp::tests/training/util_test.py::TestMakeVocabFromParams.test_exception_serialization ---
def test_exception_serialization(self):
        e = ConfigurationError("example")
        assert {"message": "example"} == vars(pickle.loads(pickle.dumps(e)))

# --- from alteryx__featuretools::featuretools/tests/entityset_tests/test_es.py::test_empty_es_pickling ---
def test_empty_es_pickling():
    es = EntitySet(id="empty")
    pkl = pickle.dumps(es)
    unpickled = pickle.loads(pkl)

    assert es.__eq__(unpickled, deep=True)

# --- from vaexio__vaex::tests/pickle_test.py::test_drop ---
def test_drop(df_file):
    df = df_file.drop('x2')
    assert len(pickle.dumps(df)) < 2000
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])

# --- from alteryx__featuretools::featuretools/tests/entityset_tests/test_es.py::test_es_pickling ---
def test_es_pickling(es):
    pkl = pickle.dumps(es)
    unpickled = pickle.loads(pkl)

    assert es.__eq__(unpickled, deep=True)
    assert not hasattr(unpickled, WW_SCHEMA_KEY)

# --- from explosion__spaCy::spacy/tests/lang/vi/test_serialize.py::test_vi_tokenizer_pickle ---
def test_vi_tokenizer_pickle(vi_tokenizer):
    b = pickle.dumps(vi_tokenizer)
    vi_tokenizer_re = pickle.loads(b)
    assert vi_tokenizer.to_bytes() == vi_tokenizer_re.to_bytes()

# --- from explosion__spaCy::spacy/tests/lang/ja/test_serialize.py::test_ja_tokenizer_pickle ---
def test_ja_tokenizer_pickle(ja_tokenizer):
    b = pickle.dumps(ja_tokenizer)
    ja_tokenizer_re = pickle.loads(b)
    assert ja_tokenizer.to_bytes() == ja_tokenizer_re.to_bytes()

# --- from modin-project__modin::modin/tests/pandas/dataframe/test_pickle.py::test_dataframe_pickle ---
def test_dataframe_pickle(request, modin_df_name):
    modin_df = request.getfixturevalue(modin_df_name)
    other = pickle.loads(pickle.dumps(modin_df))
    df_equals(modin_df, other)
