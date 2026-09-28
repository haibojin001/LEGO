# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg43::accelerate.utils.gather_object+allennlp.modules.attention.LinearAttention+allennlp.modules.seq2seq_encoders.PytorchTransformer
# name: accelerate_allennlp_primitive
# summary: Uses accelerate.utils.gather_object, allennlp.modules.attention.LinearAttention, allennlp.modules.seq2seq_encoders.PytorchTransformer, allennlp.modules.seq2vec_encoders.PytorchSeq2VecWrapper across 18 repos
# anchor_symbols: ['accelerate.utils.gather_object', 'allennlp.modules.attention.LinearAttention', 'allennlp.modules.seq2seq_encoders.PytorchTransformer', 'allennlp.modules.seq2vec_encoders.PytorchSeq2VecWrapper', 'allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm', 'allennlp.nn.util.batched_index_select']
# observed in 18 repos: ['DeepWisdom__AutoDL', 'HazyResearch__meerkat', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning', 'allenai__allennlp']...

# --- from microsoft__nni::nni/nas/hub/pytorch/modules/autoactivation.py::UnaryLogExp.forward ---
def forward(self, x):
        return torch.log(1 + torch.exp(x))

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/skeleton/nn/modules/wrappers.py::MergeSum.forward ---
def forward(self, *xs):
        return torch.sum(torch.stack(xs), dim=0)

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Image/skeleton/nn/modules/wrappers.py::MergeSum.forward ---
def forward(self, *xs):
        return torch.sum(torch.stack(xs), dim=0)

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_pytorch.py::TestPytorch.test_cste_script.MyModel.forward ---
def forward(self, x):
                return torch.zeros(x.size(0)), torch.ones((x.size(1), x.size(0)), dtype=torch.int64)

# --- from sb-ai-lab__LightAutoML::lightautoml/text/embed.py::NLinearMemoryEfficient.__init__ ---
def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

# --- from awslabs__gluonts::src/gluonts/torch/model/deep_npts/scaling.py::_one_if_too_small ---
def _one_if_too_small(x: torch.Tensor, min_value) -> torch.Tensor:
    return torch.where(
        x >= min_value, x, torch.ones(tuple(1 for _ in x.shape), dtype=x.dtype)
    )

# --- from bfortuner__ml-glossary::code/rnn.py::RNN.__init__ ---
def __init__(self, n_classes):
        super().__init__()
        self.hid_fc = nn.Linear(185, 128)
        self.out_fc = nn.Linear(185, n_classes)
        self.softmax = nn.LogSoftmax()

# --- from pykale__pykale::tests/predict/test_decode.py::DummyEncoder.forward ---
def forward(self, x):
        batch_size = x.shape[0]
        mean = torch.ones(batch_size, self.latent_dim)
        log_var = torch.zeros(batch_size, self.latent_dim)
        return mean, log_var

# --- from pykale__pykale::kale/embed/multimodal_fusion.py::Concat.forward ---
def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)

# --- from allenai__allennlp::allennlp/fairness/bias_metrics.py::EmbeddingCoherenceTest._get_ranks ---
def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(x.size(0), device=ranks.device)
        return ranks

# --- from Lightning-AI__torchmetrics::tests/unittests/wrappers/test_multioutput.py::_multi_target_sk_accuracy ---
def _multi_target_sk_accuracy(preds, target, num_outputs):
    """Compute accuracy over multiple outputs."""
    return [accuracy_score(torch.argmax(preds[:, :, i], dim=1), target[:, i]) for i in range(num_outputs)]

# --- from xorbitsai__xorbits::python/xorbits/_mars/learn/contrib/pytorch/tests/pytorch_sample.py::get_model ---
def get_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax(),
    )
