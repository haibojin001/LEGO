# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg505::torch.are_deterministic_algorithms_enabled+torch.bincount+torch.unique
# name: torch_primitive
# summary: Uses torch.are_deterministic_algorithms_enabled, torch.bincount, torch.unique across 2 repos
# anchor_symbols: ['torch.are_deterministic_algorithms_enabled', 'torch.bincount', 'torch.unique']
# observed in 2 repos: ['Lightning-AI__torchmetrics', 'freud14__poutyne']...

# --- from freud14__poutyne::poutyne/framework/metrics/predefined/bincount.py::_bincount ---
def _bincount(x: Tensor, minlength: int | None = None) -> Tensor:
    """PyTorch currently does not support``torch.bincount`` for:

        - deterministic mode on GPU.
        - MPS devices

    This implementation fallback to a for-loop counting occurrences in that case.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x
    """
    if minlength is None:
        minlength = len(torch.unique(x))
    if torch.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE or (_TORCH_GREATER_EQUAL_1_12 and x.is_mps):
        output = torch.zeros(minlength, device=x.device, dtype=torch.long)
        for i in range(minlength):
            output[i] = (x == i).sum()
        return output
    return torch.bincount(x, minlength=minlength)

# --- from Lightning-AI__torchmetrics::src/torchmetrics/utilities/data.py::_bincount ---
def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """Implement custom bincount.

    PyTorch currently does not support ``torch.bincount`` when running in deterministic mode on GPU or when running
    MPS devices or when running on XLA device. This implementation therefore falls back to using a combination of
    `torch.arange` and `torch.eq` in these scenarios. A small performance hit can expected and higher memory consumption
    as `[batch_size, mincount]` tensor needs to be initialized compared to native ``torch.bincount``.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x

    Example:
        >>> x = torch.tensor([0,0,0,1,1,2,2,2,2])
        >>> _bincount(x, minlength=3)
        tensor([3, 2, 4])

    """
    if minlength is None:
        minlength = len(torch.unique(x))

    if torch.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE or x.is_mps:
        mesh = torch.arange(minlength, device=x.device).repeat(len(x), 1)
        return torch.eq(x.reshape(-1, 1), mesh).sum(dim=0)

    return torch.bincount(x, minlength=minlength)
