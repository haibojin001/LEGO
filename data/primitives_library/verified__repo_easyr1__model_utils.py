from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn


@lru_cache
def is_rank0() -> int:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def print_gpu_memory_usage(prefix: str = "GPU memory usage") -> None:
    """Report the current GPU VRAM usage."""
    if is_rank0():
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_gb = (total_mem - free_mem) / (1024**3)
        total_gb = total_mem / (1024**3)
        print(f"{prefix}: {used_gb:.2f} GB / {total_gb:.2f} GB.")


def _get_model_size(model: nn.Module, scale: str = "auto") -> Tuple[float, str]:
    """Compute the model size."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if scale == "auto":
        if parameter_count > 1e9:
            scale = "B"
        elif parameter_count > 1e6:
            scale = "M"
        elif parameter_count > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        parameter_count = parameter_count / 1e9
    elif scale == "M":
        parameter_count = parameter_count / 1e6
    elif scale == "K":
        parameter_count = parameter_count / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}.")

    return parameter_count, scale


def print_model_size(model: nn.Module, name: Optional[str] = None) -> None:
    """Print the model size."""
    if is_rank0():
        parameter_count, scale = _get_model_size(model, scale="auto")
        if name is None:
            name = model.__class__.__name__
        print(f"{name} contains {parameter_count:.2f}{scale} parameters.")