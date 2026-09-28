# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg78::_jsonnet.evaluate_snippet+allennlp.commands.main+allennlp.common.checks.ConfigurationError
# name: _jsonnet_allennlp_primitive
# summary: Uses _jsonnet.evaluate_snippet, allennlp.commands.main, allennlp.common.checks.ConfigurationError, allennlp.common.file_utils.cached_path across 20 repos
# anchor_symbols: ['_jsonnet.evaluate_snippet', 'allennlp.commands.main', 'allennlp.common.checks.ConfigurationError', 'allennlp.common.file_utils.cached_path', 'allennlp.common.file_utils.get_file_extension', 'allennlp.common.file_utils.hardlink_or_copy']
# observed in 20 repos: ['HazyResearch__meerkat', 'HoloClean__holoclean', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning', 'OpenDCAI__DataFlex']...

# --- from allenai__allennlp::allennlp/training/metrics/auc.py::Auc.reset ---
def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

# --- from allenai__allennlp::tests/nn/checkpoint/fairscale_checkpoint_wrapper_test.py::FeedForwardForTesting.__init__ ---
def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(3, 3)

# --- from Lightning-AI__torchmetrics::tests/unittests/bases/test_metric.py::test_check_register_not_in_metric_state.TempDummyMetric.__init__ ---
def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buffer", tensor(0, dtype=torch.float))
            self.register_parameter("parameter", Parameter(tensor(0, dtype=torch.float)))

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/ulysses/utils.py::_pad_tensor ---
def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

# --- from microsoft__RD-Agent::rdagent/log/ui/utils.py::get_summary_df.compare_score ---
def compare_score(s1, s2):
        if s1 is None or s2 is None:
            return None
        try:
            c_value = math.exp(abs(math.log(s1 / s2)))
        except Exception as e:
            c_value = None
        return c_value

# --- from HazyResearch__meerkat::tests/meerkat/block/test_tensor_block.py::test_io ---
def test_io(tmpdir):
    torch.manual_seed(123)
    block = TorchBlock(torch.randn(100, 10))
    block.write(tmpdir)
    new_block = TorchBlock.read(tmpdir)

    assert isinstance(block, TorchBlock)
    assert (block.data == new_block.data).all()

# --- from Lightning-AI__torchmetrics::tests/integrations/test_lightning.py::test_metric_collection_lightning_log.TestModel.__init__ ---
def __init__(self) -> None:
            super().__init__()
            self.metric = MetricCollection([SumMetric(), DiffMetric()])
            self.register_buffer("sum", torch.tensor(0.0))
            self.register_buffer("diff", torch.tensor(0.0))

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_operators.py::TestOperators.test_basic_concat2 ---
def test_basic_concat2(self):
        class SimpleOp(nn.Module):
            def forward(self, inputs):
                out = torch.cat(inputs, 1)
                return out
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.checkExportImport(SimpleOp(), ((x, y), ))

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/trainers/sequence_parallel/utils.py::GatherLoss.backward ---
def backward(ctx, *grad_output):
        _grad = grad_output[0] * dist.get_world_size(group=ctx.process_group)
        return (
            _grad.split(ctx.scatter_shape, dim=ctx.gather_idx)[
                dist.get_rank(ctx.process_group)
            ].contiguous(),
            None,
            None,
            None,
        )

# --- from snorkel-team__snorkel::test/classification/training/test_trainer.py::create_dataloader ---
def create_dataloader(task_name="task", split="train"):
    X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
    Y = torch.ones(NUM_EXAMPLES).long()

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_pytorch.py::TestPytorch.test_squeeze_runtime_dim ---
def test_squeeze_runtime_dim(self):
        class Squeeze(nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test(Squeeze(), (d1, d4))
        self.run_test(Squeeze(), (d3, d4))

# --- from snorkel-team__snorkel::test/classification/test_multitask_classifier.py::create_dataloader ---
def create_dataloader(task_name="task", split="train", **kwargs):
    X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
    Y = torch.ones(NUM_EXAMPLES).long()

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE, **kwargs)
    return dataloader
