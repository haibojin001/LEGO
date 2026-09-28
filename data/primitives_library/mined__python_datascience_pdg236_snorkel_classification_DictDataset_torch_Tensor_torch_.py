# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg236::snorkel.classification.DictDataset+torch.Tensor+torch.equal
# name: snorkel_torch_primitive
# summary: Uses snorkel.classification.DictDataset, torch.Tensor, torch.equal across 2 repos
# anchor_symbols: ['snorkel.classification.DictDataset', 'torch.Tensor', 'torch.equal']
# observed in 2 repos: ['allenai__allennlp', 'snorkel-team__snorkel']...

# --- from snorkel-team__snorkel::test/classification/test_data.py::DatasetTest.test_classifier_dataset ---
def test_classifier_dataset(self):
        """Unit test of DictDataset"""

        x1 = [
            torch.Tensor([1]),
            torch.Tensor([1, 2]),
            torch.Tensor([1, 2, 3]),
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([1, 2, 3, 4, 5]),
        ]

        y1 = torch.Tensor([0, 0, 0, 0, 0])

        dataset = DictDataset(
            X_dict={"data1": x1}, Y_dict={"task1": y1}, name="new_data", split="train"
        )

        # Check if the dataset is correctly constructed
        self.assertTrue(torch.equal(dataset[0][0]["data1"], x1[0]))
        self.assertTrue(torch.equal(dataset[0][1]["task1"], y1[0]))
        self.assertEqual(
            repr(dataset),
            "DictDataset(name=new_data, X_keys=['data1'], Y_keys=['task1'])",
        )

        # Test from_tensors inits with default values
        dataset = DictDataset.from_tensors(x1, y1, "train")
        self.assertEqual(
            repr(dataset),
            f"DictDataset(name={DEFAULT_DATASET_NAME}, "
            f"X_keys=['{DEFAULT_INPUT_DATA_KEY}'], Y_keys=['{DEFAULT_TASK_NAME}'])",
        )

# --- from snorkel-team__snorkel::test/classification/test_utils.py::UtilsTest.test_pad_batch ---
def test_pad_batch(self):
        batch = [torch.Tensor([1, 2]), torch.Tensor([3]), torch.Tensor([4, 5, 6])]
        padded_batch, mask_batch = pad_batch(batch)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]]))
        )
        self.assertTrue(
            torch.equal(mask_batch, torch.Tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        )

        padded_batch, mask_batch = pad_batch(batch, max_len=2)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2], [3, 0], [4, 5]]))
        )
        self.assertTrue(torch.equal(mask_batch, torch.Tensor([[0, 0], [0, 1], [0, 0]])))

        padded_batch, mask_batch = pad_batch(batch, pad_value=-1)

        self.assertTrue(
            torch.equal(
                padded_batch, torch.Tensor([[1, 2, -1], [3, -1, -1], [4, 5, 6]])
            )
        )
        self.assertTrue(
            torch.equal(mask_batch, torch.Tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        )

        padded_batch, mask_batch = pad_batch(batch, left_padded=True)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[0, 1, 2], [0, 0, 3], [4, 5, 6]]))
        )
        self.assertTrue(
            torch.equal(mask_batch, torch.Tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))
        )

        padded_batch, mask_batch = pad_batch(batch, max_len=2, left_padded=True)

        self.assertTrue(
            torch.equal(padded_batch, torch.Tensor([[1, 2], [0, 3], [5, 6]]))
        )
        self.assertTrue(torch.equal(mask_batch, torch.Tensor([[0, 0], [1, 0], [0, 0]])))

# --- from allenai__allennlp::tests/modules/token_embedders/embedding_test.py::TestEmbedding.test_read_embedding_file_inside_archive ---
def test_read_embedding_file_inside_archive(self):
        token2vec = {
            "think": torch.Tensor([0.143, 0.189, 0.555, 0.361, 0.472]),
            "make": torch.Tensor([0.878, 0.651, 0.044, 0.264, 0.872]),
            "difference": torch.Tensor([0.053, 0.162, 0.671, 0.110, 0.259]),
            "àèìòù": torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        vocab = Vocabulary()
        for token in token2vec:
            vocab.add_token_to_namespace(token)

        params = Params(
            {
                "pretrained_file": str(self.FIXTURES_ROOT / "embeddings/multi-file-archive.zip"),
                "embedding_dim": 5,
            }
        )
        with pytest.raises(
            ValueError,
            match="The archive .*/embeddings/multi-file-archive.zip contains multiple files, "
            "so you must select one of the files inside "
            "providing a uri of the type: "
            "\\(path_or_url_to_archive\\)#path_inside_archive\\.",
        ):
            Embedding.from_params(params, vocab=vocab)

        for ext in [".zip", ".tar.gz"]:
            archive_path = str(self.FIXTURES_ROOT / "embeddings/multi-file-archive") + ext
            file_uri = format_embeddings_file_uri(archive_path, "folder/fake_embeddings.5d.txt")
            params = Params({"pretrained_file": file_uri, "embedding_dim": 5})
            embeddings = Embedding.from_params(params, vocab=vocab).weight.data
            for tok, vec in token2vec.items():
                i = vocab.get_token_index(tok)
                assert torch.equal(embeddings[i], vec), "Problem with format " + archive_path
