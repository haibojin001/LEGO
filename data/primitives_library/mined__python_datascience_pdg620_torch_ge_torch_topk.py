# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg620::torch.ge+torch.topk
# name: torch_primitive
# summary: Uses torch.ge, torch.topk across 2 repos
# anchor_symbols: ['torch.ge', 'torch.topk']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::GumbelTopKSampler.sample_discrete ---
def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()

        return samples

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::SoftSubSampler.sample_discrete ---
def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()

        return samples

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::GumbelTopKSampler.sample_discrete ---
def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()

        return samples

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::SoftSubSampler.sample_discrete ---
def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()

        return samples
