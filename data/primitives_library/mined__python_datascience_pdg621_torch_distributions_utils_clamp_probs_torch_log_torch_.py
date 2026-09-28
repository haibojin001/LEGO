# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg621::torch.distributions.utils.clamp_probs+torch.log+torch.rand_like
# name: torch_primitive
# summary: Uses torch.distributions.utils.clamp_probs, torch.log, torch.rand_like across 2 repos
# anchor_symbols: ['torch.distributions.utils.clamp_probs', 'torch.log', 'torch.rand_like']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::SoftSubSampler.inject_noise ---
def inject_noise(self, logits):
        u = clamp_probs(torch.rand_like(logits))
        z = -torch.log(-torch.log(u))
        noisy_logits = logits + z
        return noisy_logits

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::SoftSubSampler.inject_noise ---
def inject_noise(self, logits):
        u = clamp_probs(torch.rand_like(logits))
        z = -torch.log(-torch.log(u))
        noisy_logits = logits + z
        return noisy_logits
