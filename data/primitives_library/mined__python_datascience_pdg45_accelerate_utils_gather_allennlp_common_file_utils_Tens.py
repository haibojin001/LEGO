# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg45::accelerate.utils.gather+allennlp.common.file_utils.TensorCache+allennlp.common.file_utils.cached_path
# name: accelerate_allennlp_primitive
# summary: Uses accelerate.utils.gather, allennlp.common.file_utils.TensorCache, allennlp.common.file_utils.cached_path, allennlp.common.testing.run_distributed_test across 19 repos
# anchor_symbols: ['accelerate.utils.gather', 'allennlp.common.file_utils.TensorCache', 'allennlp.common.file_utils.cached_path', 'allennlp.common.testing.run_distributed_test', 'allennlp.fairness.bias_direction.ClassificationNormalBiasDirection', 'allennlp.fairness.bias_direction.PCABiasDirection']
# observed in 19 repos: ['AgnostiqHQ__covalent', 'BlackHC__toma', 'HazyResearch__meerkat', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning']...

# --- from microsoft__nni::nni/nas/hub/pytorch/modules/autoactivation.py::UnaryMax.forward ---
def forward(self, x):
        return torch.max(x, torch.zeros_like(x))

# --- from xorbitsai__xorbits::asv/benchmarks/serialize.py::SerializeContainersSuite.time_pickle_serialize_deserialize_list ---
def time_pickle_serialize_deserialize_list(self):
        deserialize(*cloudpickle.loads(cloudpickle.dumps(serialize(self.test_list))))

# --- from xorbitsai__xorbits::asv/benchmarks/serialize.py::SerializeContainersSuite.time_pickle_serialize_deserialize_dict ---
def time_pickle_serialize_deserialize_dict(self):
        deserialize(*cloudpickle.loads(cloudpickle.dumps(serialize(self.test_dict))))

# --- from Lightning-AI__torchmetrics::tests/unittests/bases/test_ddp.py::_test_sync_with_empty_lists ---
def _test_sync_with_empty_lists(rank):
    dummy = DummyListMetric()
    val = dummy.compute()
    assert torch.allclose(val, tensor([]))

# --- from Lightning-AI__torchmetrics::src/torchmetrics/wrappers/multioutput.py::MultioutputWrapper.compute ---
def compute(self) -> Tensor:
        """Compute metrics."""
        return torch.stack([cast(Metric, m).compute() for m in self.metrics], 0)

# --- from pykale__pykale::tests/pipeline/test_drugban_trainer.py::dummy_batch ---
def dummy_batch():
    v_d = torch.rand(4, 10)
    v_p = torch.rand(4, 10)
    labels = torch.randint(0, 2, (4,))
    return v_d, v_p, labels

# --- from pykale__pykale::tests/prepdata/test_tensor_reshape.py::test_normalize_basic ---
def test_normalize_basic():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    norm = normalize_tensor(tensor)
    assert torch.allclose(norm, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)

# --- from allenai__allennlp::benchmarks/nn/util_bench.py::bench_create_tensor_then_send_to_device ---
def bench_create_tensor_then_send_to_device(benchmark):
    device = torch.device("cuda:0")

    def create_tensor():
        return torch.rand((32, 50)).to(device)

    benchmark(create_tensor)

# --- from allenai__allennlp::benchmarks/nn/util_bench.py::bench_create_tensor_directly_on_device ---
def bench_create_tensor_directly_on_device(benchmark):
    device = torch.device("cuda:0")

    def create_tensor():
        return torch.rand((32, 50), device=device)

    benchmark(create_tensor)

# --- from awslabs__gluonts::src/gluonts/nursery/robust-mts-attack/pts/model/tft/tft_modules.py::GatedLinearUnit.forward ---
def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = torch.chunk(x, 2, dim=self.dim)
        if self.nonlinear:
            val = torch.tanh(val)
        return torch.sigmoid(gate) * val

# --- from freud14__poutyne::tests/framework/model/test_model.py::SomeDataset.__init__ ---
def __init__(self, length):
        super().__init__()
        self.length = length
        self.x = torch.rand(length, 1, 28, 28)  # Something like MNIST
        self.y = torch.randint(10, size=(length,))

# --- from microsoft__nni::nni/contrib/distillation/utils.py::_to_tensor ---
def _to_tensor(sample):
    if isinstance(sample, torch.Tensor):
        return sample
    elif isinstance(sample, numpy.ndarray):
        return torch.from_numpy(sample)
    else:
        return torch.tensor(sample)
