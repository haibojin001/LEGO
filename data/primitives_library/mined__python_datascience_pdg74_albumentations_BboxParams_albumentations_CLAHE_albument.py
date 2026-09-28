# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg74::albumentations.BboxParams+albumentations.CLAHE+albumentations.ColorJitter
# name: albumentations_primitive
# summary: Uses albumentations.BboxParams, albumentations.CLAHE, albumentations.ColorJitter, albumentations.Compose across 26 repos
# anchor_symbols: ['albumentations.BboxParams', 'albumentations.CLAHE', 'albumentations.ColorJitter', 'albumentations.Compose', 'albumentations.Emboss', 'albumentations.HorizontalFlip']
# observed in 26 repos: ['AgnostiqHQ__covalent', 'DeepWisdom__AutoDL', 'EpistasisLab__scikit-rebate', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning']...

# --- from run-house__kubetorch::python_client/tests/assets/torch_summer/torch_summer.py::torch_summer ---
def torch_summer(a, b):
    import torch

    res = int(torch.sum(torch.tensor([a, b])))
    return res

# --- from freud14__poutyne::tests/framework/model/test_model_optimizer.py::ModelOptimizerInstanciationTest.setup_method ---
def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()

# --- from microsoft__nni::test/ut/nas/nn/test_choice.py::test_add_mutable.Net.forward ---
def forward(self, x):
            if self.head:
                return torch.ones_like(x)
            else:
                return torch.zeros_like(x)

# --- from OML-Team__open-metric-learning::oml/transforms/images/albumentations.py::get_normalisation_albu ---
def get_normalisation_albu(mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    return albu.Compose([albu.Normalize(mean=mean, std=std), ToTensorV2()])

# --- from microsoft__nni::examples/nas/legacy/transformer/retiarii_transformer_demo.py::fit.generate_square_subsequent_mask ---
def generate_square_subsequent_mask(sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# --- from sb-ai-lab__LightAutoML::lightautoml/ml_algo/tabnet/utils.py::GLU_Layer.forward ---
def forward(self, x):
        """Forward-pass."""
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out

# --- from freud14__poutyne::tests/framework/callbacks/test_earlystopping.py::EarlyStoppingTest.setup_method ---
def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)

# --- from allenai__allennlp::allennlp/modules/bimpm_matching.py::BiMpmMatching.__init__.create_parameter ---
def create_parameter():  # utility function to create and initialize a parameter
            param = nn.Parameter(torch.zeros(num_perspectives, hidden_dim))
            torch.nn.init.kaiming_normal_(param)
            return param

# --- from recommenders-team__recommenders::recommenders/models/vae/standard_vae.py::StandardVAE._score ---
def _score(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass in eval mode and return numpy score matrix."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            x_bar, _, _ = self.model(x_tensor)
        return x_bar.cpu().numpy()

# --- from awslabs__gluonts::src/gluonts/torch/scaler.py::NOPScaler.__call__ ---
def __call__(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data).mean(
            dim=self.dim,
            keepdim=self.keepdim,
        )
        loc = torch.zeros_like(scale)
        return data, loc, scale

# --- from allenai__allennlp::tests/modules/token_embedders/pretrained_transformer_embedder_test.py::TestPretrainedTransformerEmbedder.test_encoder_decoder_model ---
def test_encoder_decoder_model(self):
        token_embedder = PretrainedTransformerEmbedder(
            "facebook/bart-large", sub_module="encoder"
        ).cuda()
        token_ids = torch.LongTensor([[1, 2, 3], [2, 3, 4]])
        mask = torch.ones_like(token_ids).bool()
        token_embedder(token_ids.cuda(), mask.cuda())

# --- from AgnostiqHQ__covalent::covalent/_file_transfer/strategies/s3_strategy.py::S3.upload.callable ---
def callable():
                """Upload file to remote S3 bucket."""
                import boto3

                profile = executor_profile
                region = executor_region
                s3 = boto3.Session(**get_boto_options(profile, region)).client("s3")

                s3.upload_file(from_filepath, bucket_name, to_filepath)
