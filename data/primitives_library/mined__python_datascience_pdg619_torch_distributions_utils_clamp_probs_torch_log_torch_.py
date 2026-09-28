# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg619::torch.distributions.utils.clamp_probs+torch.log+torch.max
# name: torch_primitive
# summary: Uses torch.distributions.utils.clamp_probs, torch.log, torch.max, torch.nn.functional.softmax across 2 repos
# anchor_symbols: ['torch.distributions.utils.clamp_probs', 'torch.log', 'torch.max', 'torch.nn.functional.softmax', 'torch.rand']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::GumbelTopKSampler.sample_continous ---
def sample_continous(self, logits):
        l_shape = (logits.shape[0], self.k, logits.shape[2])
        u = clamp_probs(torch.rand(l_shape, device=logits.device))
        gumbel = -torch.log(-torch.log(u))
        noisy_logits = (gumbel + logits) / self.T
        samples = F.softmax(noisy_logits, dim=-1)
        samples = torch.max(samples, dim=1)[0]

        return samples

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::GumbelTopKSampler.sample_continous ---
def sample_continous(self, logits):
        l_shape = (logits.shape[0], self.k, logits.shape[2])
        u = clamp_probs(torch.rand(l_shape, device=logits.device))
        gumbel = -torch.log(-torch.log(u))
        noisy_logits = (gumbel + logits) / self.T
        samples = F.softmax(noisy_logits, dim=-1)
        samples = torch.max(samples, dim=1)[0]

        return samples
