# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg267::tempfile.TemporaryDirectory+urllib.request.urlretrieve
# name: tempfile_urllib_primitive
# summary: Uses tempfile.TemporaryDirectory, urllib.request.urlretrieve across 2 repos
# anchor_symbols: ['tempfile.TemporaryDirectory', 'urllib.request.urlretrieve']
# observed in 2 repos: ['alteryx__featuretools', 'drivendataorg__cookiecutter-data-science']...

# --- from alteryx__featuretools::featuretools/tests/entityset_tests/test_serialization.py::test_deserialize_local_tar ---
def test_deserialize_local_tar(es):
    with tempfile.TemporaryDirectory() as tmp_path:
        temp_tar_filepath = os.path.join(tmp_path, TEST_FILE)
        urlretrieve(URL, filename=temp_tar_filepath)
        new_es = deserialize.read_entityset(temp_tar_filepath)
        assert es.__eq__(new_es, deep=True)

# --- from alteryx__featuretools::featuretools/tests/entityset_tests/test_serialization.py::test_deserialize_errors_if_python_version_unsafe ---
def test_deserialize_errors_if_python_version_unsafe(mock_inspect, es):
    mock_response = MagicMock()
    mock_response.kwonlyargs = []
    mock_inspect.return_value = mock_response
    with tempfile.TemporaryDirectory() as tmp_path:
        temp_tar_filepath = os.path.join(tmp_path, TEST_FILE)
        urlretrieve(URL, filename=temp_tar_filepath)
        with pytest.raises(RuntimeError, match=""):
            deserialize.read_entityset(temp_tar_filepath)

# --- from drivendataorg__cookiecutter-data-science::ccds/hook_utils/custom_config.py::write_custom_config ---
def write_custom_config(user_input_config):
    if not user_input_config:
        return

    tmp = TemporaryDirectory()
    tmp_zip = None

    print(user_input_config)

    # if not absolute, test if local path relative to parent of created directory
    if not user_input_config.startswith("/"):
        test_path = Path("..") / user_input_config
    else:
        test_path = Path(user_input_config)

    # check if user passed a local path
    if test_path.exists() and test_path.is_dir():
        local_path = test_path

    elif test_path.exists() and test_path.endswith(".zip"):
        tmp_zip = test_path

    # check if user passed a url to a zip
    elif user_input_config.startswith("http") and (
        user_input_config.split(".")[-1] in ["zip"]
    ):
        tmp_zip, _ = urlretrieve(user_input_config)

    # assume it is a VCS uri and try to clone
    else:
        clone(user_input_config, clone_to_dir=tmp)
        local_path = tmp

    if tmp_zip:
        with ZipFile(tmp_zip, "r") as zipf:
            zipf.extractall(tmp)
            local_path = tmp

    # write whatever the user supplied into the project
    copytree(local_path, ".")

    tmp.cleanup()
