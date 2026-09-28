# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg40::albumentations.BboxParams+albumentations.Compose+albumentations.NoOp
# name: albumentations_allennlp_primitive
# summary: Uses albumentations.BboxParams, albumentations.Compose, albumentations.NoOp, allennlp.common.Params across 20 repos
# anchor_symbols: ['albumentations.BboxParams', 'albumentations.Compose', 'albumentations.NoOp', 'allennlp.common.Params', 'allennlp.common.checks.ConfigurationError', 'allennlp.common.params.Params']
# observed in 20 repos: ['HazyResearch__meerkat', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning', 'SforAiDl__KD_Lib', 'allenai__allennlp']...

# --- from sb-ai-lab__LightAutoML::lightautoml/text/embed.py::Periodic._cos_sin ---
def _cos_sin(x: Tensor) -> Tensor:
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_operators.py::TestOperators.test_basic_basic.SimpleOp.forward ---
def forward(self, x, y):
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_operators.py::TestOperators.test_basic_params.SimpleOp.forward ---
def forward(self, x, y):
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out

# --- from deepchecks__deepchecks::tests/vision/checks/train_test_validation/property_label_correlation_change_test.py::coco_collate_with_bias_one_class.move_class ---
def move_class(tensor):
        return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) \
            if len(tensor) > 0 else tensor

# --- from deepchecks__deepchecks::deepchecks/vision/datasets/detection/coco_torch.py::collate_without_model.move_class ---
def move_class(tensor):
        return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) \
            if len(tensor) > 0 else tensor

# --- from allenai__allennlp::allennlp/nn/activations.py::GeluNew.forward ---
def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )

# --- from Lightning-AI__torchmetrics::src/torchmetrics/functional/image/lpips.py::_normalize_tensor ---
def _normalize_tensor(in_feat: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize input tensor."""
    norm_factor = torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / norm_factor

# --- from allenai__allennlp::allennlp/nn/beam_search.py::GumbelSampler.gumbel ---
def gumbel(self, phi) -> torch.Tensor:
        """
        Sample `Gumbel(phi)`.

        `phi` should have shape `(batch_size, num_classes)`.
        """
        return -torch.log(-torch.log(torch.rand_like(phi))) + phi

# --- from OML-Team__open-metric-learning::tests/test_oml/test_losses/test_triplet.py::test_soft_triplet_loss ---
def test_soft_triplet_loss() -> None:
    vec = torch.randn(32, 1024)
    gt = torch.tensor(0.693147180559945)  # this value is log1p(0)

    assert TripletLoss(margin=None, reduction="mean", need_logs=True)(vec, vec, vec).isclose(gt)

# --- from Lightning-AI__torchmetrics::tests/unittests/bases/test_ddp.py::_test_ddp_cat ---
def _test_ddp_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [tensor([1])]
    dummy._sync_dist()

    assert torch.all(torch.eq(dummy.foo, tensor([1, 1])))

# --- from awslabs__gluonts::src/gluonts/nursery/daf/tslib/nn/activations.py::GeLU.forward ---
def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1
                + pt.tanh(
                    math.sqrt(2 / math.pi) * (x + 0.044715 * pt.pow(x, 3))
                )
            )
        )

# --- from awslabs__gluonts::src/gluonts/nursery/daf/tslib/nn/activations.py::GatedLinearUnit.forward ---
def forward(self, x: Tensor) -> Tensor:
        if x.size(self.dim) % 2 > 0:
            raise ValueError("The specified dimension must be even")
        val, gate = x.chunk(2, dim=self.dim)
        if self.nonlinear:
            val = pt.tanh(val)
        gate = pt.sigmoid(gate)
        return val * gate
