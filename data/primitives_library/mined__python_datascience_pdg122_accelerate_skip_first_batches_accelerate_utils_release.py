# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg122::accelerate.skip_first_batches+accelerate.utils.release_memory+allennlp.common.params.Params
# name: accelerate_allennlp_primitive
# summary: Uses accelerate.skip_first_batches, accelerate.utils.release_memory, allennlp.common.params.Params, allennlp.common.testing.run_distributed_test across 14 repos
# anchor_symbols: ['accelerate.skip_first_batches', 'accelerate.utils.release_memory', 'allennlp.common.params.Params', 'allennlp.common.testing.run_distributed_test', 'allennlp.data.dataset_readers.SequenceTaggingDatasetReader', 'allennlp.data.dataset_readers.ShardedDatasetReader']
# observed in 14 repos: ['Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning', 'OpenDCAI__DataFlex', 'allenai__allennlp', 'awslabs__gluonts']...

# --- from pykale__pykale::tests/pipeline/test_base_nn_trainer.py::test_test_step ---
def test_test_step(multimodal_model):
    x = [torch.rand(1, 5) for _ in range(2)]
    y = torch.tensor([1])
    batch = [*x, y]
    multimodal_model.test_step(batch, 0)

# --- from pykale__pykale::tests/pipeline/test_base_nn_trainer.py::test_validation_step ---
def test_validation_step(multimodal_model):
    x = [torch.rand(1, 5) for _ in range(2)]
    y = torch.tensor([1])
    batch = [*x, y]
    multimodal_model.validation_step(batch, 0)

# --- from OML-Team__open-metric-learning::tests/test_oml/test_datasets/test_datasets.py::ASCITokenizer.tokenize ---
def tokenize(text: str, *args, **kwargs):  # type: ignore
        ids = LongTensor(list(map(ord, text)))
        data = {"input_ids": ids, "attention_mask": torch.ones(len(ids)).long()}
        return data

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_pytorch.py::TestPytorch.test_slice_with_input_index ---
def test_slice_with_input_index(self):
        class InputIndexSlice(nn.Module):
            def forward(self, x, y):
                x[:y.size(0), 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

# --- from OML-Team__open-metric-learning::tests/test_integrations/test_lightning/test_pipeline.py::DummyRetrievalDataset.__getitem__ ---
def __getitem__(self, item: int) -> Dict[str, Any]:
        input_tensors = torch.rand((3, self.im_size, self.im_size))
        label = torch.tensor(self.labels[item]).long()
        return {
            INPUT_TENSORS_KEY: input_tensors,
            LABELS_KEY: label,
            INDEX_KEY: item,
        }

# --- from allenai__allennlp::tests/training/metrics/f1_measure_test.py::F1MeasureTest.test_f1_measure_catches_exceptions ---
def test_f1_measure_catches_exceptions(self, device: str):
        f1_measure = F1Measure(0)
        predictions = torch.rand([5, 7], device=device)
        out_of_range_labels = torch.tensor([10, 3, 4, 0, 1], device=device)
        with pytest.raises(ConfigurationError):
            f1_measure(predictions, out_of_range_labels)

# --- from allenai__allennlp::tests/nn/util_test.py::TestNnUtil.test_get_mask_from_sequence_lengths ---
def test_get_mask_from_sequence_lengths(self):
        sequence_lengths = torch.LongTensor([4, 3, 1, 4, 2])
        mask = util.get_mask_from_sequence_lengths(sequence_lengths, 5).data.numpy()
        assert_almost_equal(
            mask,
            [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]],
        )

# --- from Lightning-AI__torchmetrics::tests/unittests/classification/test_average_precision.py::test_warning_on_no_positives ---
def test_warning_on_no_positives():
    """Test that a warning is raised when there are no positive samples in the target."""
    preds = torch.rand(100)
    target = torch.zeros(100).long()
    with pytest.warns(UserWarning, match="No positive samples found in target, recall is undefined. Setting recall.*"):
        binary_average_precision(preds, target)

# --- from encord-team__encord-active::src/encord_active/lib/metrics/semantic/_class_uncertainty.py::base_predict ---
def base_predict(classifier, batch, name_to_idx):
    embeddings = torch.Tensor([eval(x) for x in batch["embedding"]]).to(DEVICE)
    labels = torch.LongTensor([name_to_idx[x] for x in batch["object_class"]]).to(DEVICE)
    outputs = classifier(embeddings)
    model_preds, mean_probs, entropy = get_prediction_statistics(outputs)
    return labels, mean_probs, model_preds, entropy

# --- from snorkel-team__snorkel::test/classification/test_loss.py::SoftCrossEntropyTest.test_invalid_reduction ---
def test_invalid_reduction(self):
        Y_golds = torch.LongTensor([0, 1, 2])
        Y_golds_probs = torch.Tensor(preds_to_probs(Y_golds.numpy(), num_classes=4))

        Y_probs = torch.rand_like(Y_golds_probs)
        Y_probs = Y_probs / Y_probs.sum(dim=1).reshape(-1, 1)

        with self.assertRaisesRegex(ValueError, "Keyword 'reduction' must be"):
            cross_entropy_with_probs(Y_probs, Y_golds_probs, reduction="bad")

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/utils/utils.py::add_version_to_work_dir ---
def add_version_to_work_dir(work_dir: str) -> str:
    """add version"""
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sub_folder = f"v{version}-{time}"
    if (dist.is_initialized() and is_dist()) or is_dist_ta():
        obj_list = [sub_folder]
        dist.broadcast_object_list(obj_list)
        sub_folder = obj_list[0]

    work_dir = os.path.join(work_dir, sub_folder)
    return work_dir

# --- from snorkel-team__snorkel::test/classification/test_loss.py::SoftCrossEntropyTest.test_perfect_predictions ---
def test_perfect_predictions(self):
        # Does soft ce loss achieve approx. 0 loss with perfect predictions?
        Y_golds = torch.LongTensor([0, 1, 2])
        Y_golds_probs = torch.Tensor(preds_to_probs(Y_golds.numpy(), num_classes=4))

        Y_probs = Y_golds_probs.clone()
        Y_probs[Y_probs == 1] = 100
        Y_probs[Y_probs == 0] = -100

        ces_loss = cross_entropy_with_probs(Y_probs, Y_golds_probs)
        np.testing.assert_equal(ces_loss.numpy(), 0)
