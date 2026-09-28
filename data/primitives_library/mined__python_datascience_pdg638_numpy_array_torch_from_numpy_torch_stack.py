# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg638::numpy.array+torch.from_numpy+torch.stack
# name: numpy_torch_primitive
# summary: Uses numpy.array, torch.from_numpy, torch.stack across 2 repos
# anchor_symbols: ['numpy.array', 'torch.from_numpy', 'torch.stack']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/text/utils.py::custom_collate ---
def custom_collate(batch: List[np.ndarray]) -> torch.Tensor:
    """Puts each data field into a tensor with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    else:
        return torch.from_numpy(np.array(batch)).float()

# --- from sberbank-ai-lab__LightAutoML::lightautoml/text/utils.py::custom_collate ---
def custom_collate(batch: List[np.ndarray]) -> torch.Tensor:
    """Puts each data field into a tensor with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    else:
        return torch.from_numpy(np.array(batch)).float()
