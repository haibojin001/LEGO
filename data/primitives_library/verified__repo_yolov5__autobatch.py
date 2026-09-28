"""Utilities for selecting a training batch size automatically."""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile, smart_amp_autocast


def check_train_batch_size(model, imgsz=640, amp=True):
    """Return an automatically estimated training batch size for a model."""
    with smart_amp_autocast(amp):
        cloned_model = deepcopy(model)
        cloned_model.train()
        return autobatch(cloned_model, imgsz)


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    """Estimate a CUDA training batch size within the requested memory fraction."""
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for --imgsz {imgsz}")

    device = next(model.parameters()).device
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size

    if torch.backends.cudnn.benchmark:
        LOGGER.warning(f"{prefix}Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    bytes_per_gib = 1 << 30
    device_name = str(device).upper()
    properties = torch.cuda.get_device_properties(device)
    total = properties.total_memory / bytes_per_gib
    reserved = torch.cuda.memory_reserved(device) / bytes_per_gib
    allocated = torch.cuda.memory_allocated(device) / bytes_per_gib
    free = total - reserved - allocated

    LOGGER.info(
        f"{prefix}{device_name} ({properties.name}) {total:.2f}G total, "
        f"{reserved:.2f}G reserved, {allocated:.2f}G allocated, {free:.2f}G free"
    )

    candidates = [1, 2, 4, 8, 16]
    try:
        images = [torch.empty(n, 3, imgsz, imgsz) for n in candidates]
        profiled = profile(images, model, n=3, device=device)
    except Exception as error:
        LOGGER.warning(f"{prefix}{error}")
        return batch_size

    memory = [entry[2] for entry in profiled if entry]
    coefficients = np.polyfit(candidates[: len(memory)], memory, deg=1)
    selected = int((free * fraction - coefficients[1]) / coefficients[0])

    if None in profiled:
        first_failure = profiled.index(None)
        if selected >= candidates[first_failure]:
            selected = candidates[max(first_failure - 1, 0)]

    if selected < 1 or selected > 1024:
        selected = batch_size
        LOGGER.warning(f"{prefix}CUDA anomaly detected, recommend restart environment and retry command.")

    estimated_fraction = (np.polyval(coefficients, selected) + reserved + allocated) / total
    LOGGER.info(
        f"{prefix}Using batch-size {selected} for {device_name} "
        f"{total * estimated_fraction:.2f}G/{total:.2f}G ({estimated_fraction * 100:.0f}%) ✅"
    )
    return selected