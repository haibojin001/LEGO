# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg439::pytest.warns+unittest.mock.Mock
# name: pytest_unittest_primitive
# summary: Uses pytest.warns, unittest.mock.Mock across 2 repos
# anchor_symbols: ['pytest.warns', 'unittest.mock.Mock']
# observed in 2 repos: ['ottogroup__palladium', 'ploomber__ploomber']...

# --- from ploomber__ploomber::tests/dag/test_dagqualitychecker.py::test_warn_on_sql_missing_docstrings ---
def test_warn_on_sql_missing_docstrings():
    dag = DAG()

    sql = "SELECT * FROM table"
    SQLDump(sql, File("file1.txt"), dag, client=Mock(), name="sql")

    qc = DAGQualityChecker()

    with pytest.warns(UserWarning):
        qc(dag)

# --- from ploomber__ploomber::tests/dag/test_dagqualitychecker.py::test_does_not_warn_on_sql_docstrings ---
def test_does_not_warn_on_sql_docstrings():
    dag = DAG()

    sql = "/* get data from table */\nSELECT * FROM table"
    SQLDump(sql, File("file1.txt"), dag, client=Mock(), name="sql")

    qc = DAGQualityChecker()

    with pytest.warns(None) as warn:
        qc(dag)

    assert not warn

# --- from ottogroup__palladium::palladium/tests/test_fit.py::TestGridSearch.test_deprecated_scoring ---
def test_deprecated_scoring(self, grid_search, GridSearchCVWithScores):
        # 'scoring' inside of 'grid_search' is deprecated
        model = Mock(spec=['fit', 'predict', 'score'])
        dataset_loader_train = Mock()
        scoring = Mock()
        dataset_loader_train.return_value = object(), object()

        with pytest.warns(DeprecationWarning):
            grid_search(dataset_loader_train, model,
                        {'scoring': scoring}, scoring=None)
        GridSearchCVWithScores.assert_called_with(
            model, refit=False, scoring=scoring)
