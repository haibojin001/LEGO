# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg124::allennlp.nn.util.dist_reduce_sum+functools.reduce+math.ceil
# name: allennlp_functools_primitive
# summary: Uses allennlp.nn.util.dist_reduce_sum, functools.reduce, math.ceil, math.log2 across 26 repos
# anchor_symbols: ['allennlp.nn.util.dist_reduce_sum', 'functools.reduce', 'math.ceil', 'math.log2', 'mmdet.apis.init_detector', 'mmdet.testing.demo_mm_inputs']
# observed in 26 repos: ['AgnostiqHQ__covalent', 'BiomedSciAI__causallib', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning', 'OpenDCAI__DataFlex']...

# --- from microsoft__nni::nni/nas/hub/pytorch/modules/autoactivation.py::UnaryExpSquare.forward ---
def forward(self, x):
        return torch.exp(-torch.square(x))

# --- from microsoft__nni::nni/nas/hub/pytorch/modules/autoactivation.py::UnaryLogAbs.forward ---
def forward(self, x):
        return torch.log(torch.abs(x) + 1e-7)

# --- from allenai__allennlp::allennlp/nn/regularizers/regularizers.py::L1Regularizer.__call__ ---
def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))

# --- from allenai__allennlp::allennlp/nn/regularizers/regularizers.py::L2Regularizer.__call__ ---
def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))

# --- from SforAiDl__KD_Lib::KD_Lib/models/shallow.py::Shallow.forward ---
def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        return out

# --- from bfortuner__ml-glossary::code/vae.py::vae_loss ---
def vae_loss(output, input, mean, logvar, loss_func):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    return recon_loss + kl_loss

# --- from awslabs__gluonts::src/gluonts/nursery/daf/network/kernel.py::RBFKernel._kernel ---
def _kernel(
        self,
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        q = q.unsqueeze(dim=-2)
        k = k.unsqueeze(dim=-3)
        score = -pt.sum(pt.pow(q - k, 2), dim=-1) / self.bandwidth
        return score

# --- from WecoAI__aideml::sample_results/digit-recognizer.py::Net.forward ---
def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# --- from awslabs__gluonts::src/gluonts/nursery/robust-mts-attack/pts/modules/flows.py::Flow.inverse ---
def inverse(self, u, cond):
        x, log_abs_det_jacobian = self.net.inverse(u, cond)
        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += torch.log(torch.abs(self.scale))
        return x, log_abs_det_jacobian

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/experiment/templates/digit-recognizer/model/model_nn.py::NeuralNetwork.forward ---
def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# --- from sematic-ai__sematic::sematic/examples/cifar_classifier/train_eval.py::Net.forward ---
def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- from AgnostiqHQ__covalent::tests/stress_tests/scripts/tasks.py::NeuralNetwork.forward ---
def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
