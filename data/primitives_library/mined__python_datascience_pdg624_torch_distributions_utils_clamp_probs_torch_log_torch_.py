# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg624::torch.distributions.utils.clamp_probs+torch.log+torch.mean
# name: torch_primitive
# summary: Uses torch.distributions.utils.clamp_probs, torch.log, torch.mean, torch.sum across 2 repos
# anchor_symbols: ['torch.distributions.utils.clamp_probs', 'torch.log', 'torch.mean', 'torch.sum']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/utils.py::cross_entropy_multiple_class ---
def cross_entropy_multiple_class(input: torch.FloatTensor, target: torch.FloatTensor) -> torch.Tensor:
    """Cross entropy evaluation."""
    return torch.mean(torch.sum(-target * torch.log(clamp_probs(input)), dim=1))

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/utils.py::cross_entropy_multiple_class ---
def cross_entropy_multiple_class(input: torch.FloatTensor, target: torch.FloatTensor) -> torch.Tensor:
    """Cross entropy evaluation."""
    return torch.mean(torch.sum(-target * torch.log(clamp_probs(input)), dim=1))
