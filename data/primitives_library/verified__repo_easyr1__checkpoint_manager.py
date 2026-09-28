import json
import os
import random
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin


CHECKPOINT_TRACKER = "checkpoint_tracker.json"


class BaseCheckpointManager(ABC):
    """
    Base interface for distributed checkpoint managers.

    Implementations are responsible for persisting and restoring model,
    optimizer, scheduler, and optional auxiliary state in an SPMD setting.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class

        assert isinstance(model, FSDP)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def local_mkdir(path: str) -> str:
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        lock_name = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_file = os.path.join(tempfile.gettempdir(), lock_name)

        try:
            with FileLock(lock_file, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            os.makedirs(path, exist_ok=True)

        return path

    @staticmethod
    def get_rng_state() -> dict[str, Any]:
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    @staticmethod
    def load_rng_state(rng_state: dict[str, Any]):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])


def get_checkpoint_tracker_filename(root_path: str) -> str:
    """
    Return the location of the checkpoint tracker stored under root_path.
    """
    return os.path.join(root_path, CHECKPOINT_TRACKER)


def find_latest_ckpt(
    path: str, directory_format: str = "global_step_{}"
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Read the checkpoint tracker and locate the checkpoint it identifies.
    """
    tracker_path = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_path):
        return None, None

    with open(tracker_path, "rb") as file:
        tracker = json.load(file)

    checkpoint_path = os.path.join(
        path, directory_format.format(tracker["last_global_step"])
    )
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint does not exist: {checkpoint_path}")
        return None, None

    print(
        f"Found latest checkpoint: {checkpoint_path}, will resume from it. "
        "Turn off `find_last_checkpoint` to disable it."
    )
    return checkpoint_path, tracker


def remove_obsolete_ckpt(
    path: str,
    global_step: int,
    best_global_step: int,
    save_limit: int = -1,
    directory_format: str = "global_step_{}",
):
    """
    Delete older checkpoint directories once the configured retention limit
    has been exceeded, while preserving the designated best checkpoint.
    """
    if save_limit <= 0 or not os.path.exists(path):
        return

    keep_count = save_limit - 1
    expression = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    previous_steps = []

    for name in os.listdir(path):
        match = re.match(expression, name)
        if match:
            step = int(match.group(1))
            if step < global_step:
                previous_steps.append(step)

    previous_steps.sort(reverse=True)

    if best_global_step in previous_steps:
        previous_steps.remove(best_global_step)
        keep_count = max(keep_count - 1, 0)

    for step in previous_steps[keep_count:]:
        checkpoint_path = os.path.join(path, directory_format.format(step))
        try:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            print(f"Removed obsolete checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to remove {checkpoint_path}: {e}")