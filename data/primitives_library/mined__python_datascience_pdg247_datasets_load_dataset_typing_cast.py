# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg247::datasets.load_dataset+typing.cast
# name: datasets_typing_primitive
# summary: Uses datasets.load_dataset, typing.cast across 2 repos
# anchor_symbols: ['datasets.load_dataset', 'typing.cast']
# observed in 2 repos: ['cleanlab__cleanlab', 'deepchecks__deepchecks']...

# --- from deepchecks__deepchecks::tests/nlp/conftest.py::original_wikiann ---
def original_wikiann():
    return t.cast(t.Any, load_dataset('wikiann', name='en'))

# --- from cleanlab__cleanlab::cleanlab/datalab/internal/data.py::Data._load_dataset_from_string ---
def _load_dataset_from_string(data_string: str) -> Dataset:
        if not os.path.exists(data_string):
            try:
                dataset = datasets.load_dataset(data_string)
                return cast(Dataset, dataset)
            except Exception as error:
                raise DatasetLoadError(str) from error

        factory: Dict[str, Callable[[str], Any]] = {
            ".txt": Dataset.from_text,
            ".csv": Dataset.from_csv,
            ".json": Dataset.from_json,
        }

        extension = os.path.splitext(data_string)[1]
        if extension not in factory:
            raise DatasetLoadError(type(data_string))

        dataset = factory[extension](data_string)
        dataset_cast = cast(Dataset, dataset)
        return dataset_cast

# --- from cleanlab__cleanlab::tests/datalab/test_data.py::TestData.test_load_dataset_from_string ---
def test_load_dataset_from_string(self, monkeypatch):
        # Test with non-existent file
        with pytest.raises(DatasetLoadError):
            Data._load_dataset_from_string("non_existent_file.txt")

        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix=".invalid") as temp_file:
            with pytest.raises(DatasetLoadError):
                Data._load_dataset_from_string(temp_file.name)

        # Test with invalid external dataset identifier
        with patch("datasets.load_dataset") as mock_load_dataset:
            mock_load_dataset.side_effect = ValueError("Invalid external dataset identifier")
            with pytest.raises(DatasetLoadError) as excinfo:
                Data._load_dataset_from_string("invalid_external_dataset_name")

            expected_error_substring = "Failed to load dataset from <class 'str'>.\n"
            assert expected_error_substring in str(excinfo.value)

        # Test with valid .txt, .csv, and .json files
        test_data = [
            (".txt", "sample text", "from_text"),
            (".csv", "column1,column2\nvalue1,value2", "from_csv"),
            (".json", '{"key": "value"}', "from_json"),
        ]

        mock_dataset = Dataset.from_dict({"y": [1, 2, 3]})
        for ext, content, loader_func in test_data:
            with tempfile.NamedTemporaryFile(suffix=ext, mode="w+t") as temp_file:
                temp_file.write(content)
                temp_file.flush()

                # Make sure the correct loader function is called
                def fake_loader(file_name):
                    assert file_name == temp_file.name
                    return mock_dataset

                with monkeypatch.context() as mp:
                    mp.setattr(Dataset, loader_func, fake_loader)
                    loaded_dataset = Data._load_dataset_from_string(temp_file.name)
                    assert isinstance(loaded_dataset, Dataset)
                    assert loaded_dataset == mock_dataset

        # Test with an external dataset
        def fake_load_dataset(data_string):
            if data_string == "external_dataset":
                return mock_dataset

            raise Exception("Not the expected dataset string")

        with monkeypatch.context() as mp:
            mp.setattr("datasets.load_dataset", fake_load_dataset)
            loaded_dataset = Data._load_dataset_from_string("external_dataset")
            assert isinstance(loaded_dataset, Dataset)
            assert loaded_dataset == mock_dataset

            with pytest.raises(DatasetLoadError) as excinfo:
                Data._load_dataset_from_string("non_external_dataset")

            expected_error_substring = "Failed to load dataset from <class 'str'>.\n"
