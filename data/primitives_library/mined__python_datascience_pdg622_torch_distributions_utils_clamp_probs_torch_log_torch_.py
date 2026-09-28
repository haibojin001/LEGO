# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg622::torch.distributions.utils.clamp_probs+torch.log+torch.nn.functional.softmax
# name: torch_primitive
# summary: Uses torch.distributions.utils.clamp_probs, torch.log, torch.nn.functional.softmax, torch.stack across 2 repos
# anchor_symbols: ['torch.distributions.utils.clamp_probs', 'torch.log', 'torch.nn.functional.softmax', 'torch.stack', 'torch.zeros_like']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::SoftSubSampler.continuous_topk ---
def continuous_topk(self, w, separate=False):
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype=torch.float32)
        for _ in range(self.k):
            khot_mask = clamp_probs(1.0 - onehot_approx)
            w += torch.log(khot_mask)
            onehot_approx = F.softmax(w / self.T, dim=-1)
            khot_list.append(onehot_approx)
        if separate:
            return khot_list
        else:
            return torch.stack(khot_list, dim=-1).sum(-1).squeeze(1)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x_model.py::SoftSubSampler.continuous_topk ---
def continuous_topk(self, w, separate=False):
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype=torch.float32)
        for _ in range(self.k):
            khot_mask = clamp_probs(1.0 - onehot_approx)
            w += torch.log(khot_mask)
            onehot_approx = F.softmax(w / self.T, dim=-1)
            khot_list.append(onehot_approx)
        if separate:
            return khot_list
        else:
            return torch.stack(khot_list, dim=-1).sum(-1).squeeze(1)
