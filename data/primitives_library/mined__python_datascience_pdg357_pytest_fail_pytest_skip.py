# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg357::pytest.fail+pytest.skip
# name: pytest_primitive
# summary: Uses pytest.fail, pytest.skip across 4 repos
# anchor_symbols: ['pytest.fail', 'pytest.skip']
# observed in 4 repos: ['NannyML__nannyml', 'awslabs__gluonts', 'nok__sklearn-porter', 'xorbitsai__xorbits']...

# --- from NannyML__nannyml::tests/io/test_writers.py::test_pickle_file_writer_raises_no_exceptions_when_writing ---
def test_pickle_file_writer_raises_no_exceptions_when_writing(result):  # noqa: D103
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PickleFileWriter(path=tmpdir)
            writer.write(result, filename='export.pkl')
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")

# --- from NannyML__nannyml::tests/io/test_writers.py::test_raw_files_writer_raises_no_exceptions_when_writing_to_parquet ---
def test_raw_files_writer_raises_no_exceptions_when_writing_to_parquet(result):  # noqa: D103
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RawFilesWriter(path=tmpdir)
            writer.write(result, filename='export.pq')
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")

# --- from xorbitsai__xorbits::python/xorbits/numpy/mars_adapters/tests/test_numpy_examples.py::test_docstrings ---
def test_docstrings(setup, doctest_namespace, obj, name):
    if name in SKIPPED_NAMES:
        pytest.skip("Skipping these name as they do not have __code__")
    results = run_docstring(
        getattr(obj, name),
        doctest_namespace,
        name=name,
        verbose=True,
        optionflags=doctest.NORMALIZE_WHITESPACE,
    )

    for result in results:
        if result.failed != 0:
            pytest.fail(f"{result.failed} out of {result.attempted} example(s) failed.")
        else:
            print(f"{result.attempted} example(s) passed.")

# --- from xorbitsai__xorbits::python/xorbits/pandas/mars_adapters/tests/test_pandas_examples.py::test_docstrings ---
def test_docstrings(setup, doctest_namespace, obj, name):
    # TODO: remove this condition in further version
    if name == "median":
        pytest.skip(f"Skip name={name} due to pandas 2.1.0 doc issue.")

    results = run_docstring(
        getattr(obj, name),
        doctest_namespace,
        name=name,
        verbose=True,
        optionflags=doctest.NORMALIZE_WHITESPACE,
    )

    for result in results:
        if result.failed != 0:
            pytest.fail(f"{result.failed} out of {result.attempted} example(s) failed.")
        else:
            print(f"{result.attempted} example(s) passed.")
            print(f"{result} + is passed don't worry")

# --- from awslabs__gluonts::test/mx/test_mx_util.py::test_symb_block_import_backward_compatible ---
def test_symb_block_import_backward_compatible(block_type) -> None:
    x1 = mx.nd.array([1, 2, 3])
    x2 = [mx.nd.array([1, 5, 5]), mx.nd.array([2, 3, 3])]

    my_block = block_type()
    my_block.collect_params().initialize()
    my_block.hybridize()
    my_block(x1, x2)

    with tempfile.TemporaryDirectory(
        prefix="gluonts-estimator-temp-"
    ) as temp_dir:
        temp_path = Path(temp_dir)

        export_symb_block(my_block, temp_path, "gluonts-model")

        format_json_path = temp_path / "gluonts-model-in_out_format.json"

        assert format_json_path.exists()
        try:
            format_json_path.unlink()
            import_symb_block(3, temp_path, "gluonts-model")
        except FileNotFoundError:
            pytest.fail(
                "Symbol block import fails when format json is not in path"
            )

# --- from nok__sklearn-porter::tests/estimator/KNeighborsClassifierTest.py::test_estimator_k_neighbors_classifier ---
def test_estimator_k_neighbors_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_neighbors: int,
):
    """Test and compare different k-Neighbors classifiers."""
    orig_est = KNeighborsClassifierClass(n_neighbors=n_neighbors)

    if not can(orig_est, language, template, 'predict'):
        pytest.skip(
            'Skip unsupported estimator/'
            'language/template combination'
        )

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir, [
            ('test', 'estimator_decision_tree_classifier'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
            ('n_neighbors', str(n_neighbors)),
        ]
    )

    try:
        score = est.test(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False
        )
    except exception.CodeTooLarge:
        msg = 'Code too large for the combination: ' \
              'language: {}, template: {}, dataset: {}' \
              ''.format(language, template, dataset.name)
        warnings.warn(msg)
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))
    else:
        assert score == 1.

# --- from nok__sklearn-porter::tests/estimator/RandomForestClassifierTest.py::test_estimator_random_forest_classifier ---
def test_estimator_random_forest_classifier(
    tmp_root_dir: Path,
    dataset: Dataset,
    template: str,
    language: str,
    n_estimators: int,
    max_depth: int,
):
    """Test and compare different RandomForest classifiers."""
    orig_est = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=1
    )

    if not can(orig_est, language, template, 'predict'):
        pytest.skip(
            'Skip unsupported estimator/'
            'language/template combination'
        )

    # Estimator:
    x, y = dataset.data, dataset.target
    orig_est.fit(X=x, y=y)
    est = Estimator(orig_est)

    # Samples:
    test_x = np.vstack((dataset_uniform_x(x), dataset_generate_x(x)))

    # Directory:
    tmp_dir = fs_mkdir(
        tmp_root_dir, [
            ('test', 'estimator_decision_tree_classifier'),
            ('dataset', dataset.name), ('language', language),
            ('template', template), ('n_estimators', str(n_estimators)),
            ('max_depth', str(max_depth))
        ]
    )

    try:
        score = est.test(
            test_x,
            language=language,
            template=template,
            directory=tmp_dir,
            delete_created_files=False
        )
    except exception.CodeTooLarge:
        msg = 'Code too large for the combination: ' \
              'language: {}, template: {}, dataset: {}' \
              ''.format(language, template, dataset.name)
        warnings.warn(msg)
    except Exception as e:
        pytest.fail('Unexpected exception ... ' + str(e))
    else:
        assert score == 1.
