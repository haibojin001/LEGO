# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg146::random.seed+torch.manual_seed
# name: random_torch_primitive
# summary: Uses random.seed, torch.manual_seed across 17 repos
# anchor_symbols: ['random.seed', 'torch.manual_seed']
# observed in 17 repos: ['DeepWisdom__AutoDL', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning', 'SforAiDl__KD_Lib', 'allenai__allennlp']...

# --- from snorkel-team__snorkel::test/classification/test_multitask_classifier.py::ClassifierTest.setUp ---
def setUp(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

# --- from snorkel-team__snorkel::test/slicing/test_slice_combiner.py::SliceCombinerTest.setUpClass ---
def setUpClass(cls):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

# --- from SforAiDl__KD_Lib::KD_Lib/KD/text/BERT2LSTM/utils.py::set_seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- from microsoft__RD-Agent::test/notebook/testfiles/main2.py::seed_everything ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- from SforAiDl__KD_Lib::KD_Lib/KD/text/BERT2LSTM/bert2lstm.py::BERT2LSTM.set_seed ---
def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/deepspeed_strategy.py::DeepspeedStrategy.set_seed ---
def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/fsdp_strategy.py::FSDPStrategy.set_seed ---
def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- from Lightning-AI__torchmetrics::tests/unittests/_helpers/__init__.py::seed_all ---
def seed_all(seed):
    """Set the seed of all computational frameworks."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- from microsoft__nni::examples/nas/legacy/cdarts/utils.py::reset_seed ---
def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# --- from encord-team__encord-active::examples/maskrcnn-example/utils/provider.py::setup_reproducibility ---
def setup_reproducibility(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# --- from yzhao062__pyod::pyod/models/base_dl.py::BaseDeepLearningDetector._set_seed ---
def _set_seed(random_state):
        """Set random seed for reproducibility
        """
        os.environ['PYTHONHASHSEED'] = str(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

# --- from Lightning-AI__torchmetrics::src/conftest.py::reset_random_seed ---
def reset_random_seed(seed: int = 42) -> None:
        """Reset the random seed before running each doctest."""
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
