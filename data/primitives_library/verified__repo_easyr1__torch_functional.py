import math
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .torch_dtypes import PrecisionType

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


@torch.compiler.disable()
def log_probs_from_logits_flash_attn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses_and_z_losses = cross_entropy_loss(logits, labels, inplace_backward=True)
    if not isinstance(losses_and_z_losses, tuple):
        raise ValueError(
            "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
        )
    return -losses_and_z_losses[0]


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probs on the label ids given logits.

    We may use torch compile to speed up computing.

    Args:
        logits (torch.Tensor): logits of the model, shape (batch_size, seqlen, vocab_size)
        labels (torch.Tensor): labels of the model, shape (batch_size, seqlen)

    Returns:
        torch.Tensor: log probs of the labels, shape (batch_size, seqlen)
    """
    shape = logits.shape[:-1]
    vocabulary_size = logits.shape[-1]
    flat_logits = logits.contiguous().view(-1, vocabulary_size)
    flat_labels = labels.contiguous().view(-1)

    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        log_probabilities = log_probs_from_logits_flash_attn(flat_logits, flat_labels)
    else:
        log_probabilities = -F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")

    return log_probabilities.view(*shape)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = None, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    average = masked_mean(values, mask)
    variance = masked_mean((values - average) ** 2, mask)

    if unbiased:
        number_of_selected_values = mask.sum()
        if number_of_selected_values <= 1:
            print("The sum of the mask is less than one, which can cause a division by zero.")
            return variance
        variance = variance * number_of_selected_values / (number_of_selected_values - 1)

    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whiten values with masked values."""
    average = masked_mean(values, mask)
    variance = masked_var(values, mask)
    return (values - average) * torch.rsqrt(variance + eps)


def get_response_mask(
    response_ids: torch.Tensor, eos_token_id: Union[int, List[int]] = 2, dtype: torch.dtype = torch.long
):
    """Get the mask for the response ids, the mask will be 0 after the first eos token.

    eos_token_id can be int or list: 1 or [1, 2].
    ```
    e.g. eos_token = 1
    response_ids:  [0, 0, 2, 4, 3, 5, 1, 0, 0]
    response_mask: [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ```
    """
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_positions = torch.zeros_like(response_ids, dtype=torch.bool)
    for eos_id in eos_token_id:
        eos_positions |= response_ids.eq(eos_id)

    eos_positions = eos_positions.long()
    after_eos = (torch.cumsum(eos_positions, dim=1) - eos_positions).bool()
    return torch.logical_not(after_eos).to(dtype)


def pad_2d_list_to_length(
    response: List[List[int]], pad_token_id: int, max_length: Optional[int] = None
) -> torch.Tensor:
    """Pad a 2D list (e.g. responses, log_probs) to a 2D tensor."""
    longest = max(len(item) for item in response)
    desired_length = max_length if max_length is not None and max_length > longest else longest
    rows = [tuple(item) + (pad_token_id,) * (desired_length - len(item)) for item in response]
    return torch.tensor(rows)


def pad_sequence_to_length(
    tensor: torch.Tensor, max_seq_len: int, pad_token_id: int, left_pad: bool = False
) -> torch.Tensor:
    """Pad a nD tensors in the last dim to max_seq_len."""
    if tensor.size(-1) >= max_seq_len:
        return tensor

    padding_shape = list(tensor.shape)
    padding_shape[-1] = max_seq_len - tensor.size(-1)
    padding = torch.full(
        padding_shape,
        fill_value=pad_token_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    if left_pad:
        return torch.cat((padding, tensor), dim=-1)
    return torch.cat((tensor, padding), dim=-1)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad: bool = True,
    truncation: Literal["left", "right", "error"] = "error",
):
    """Pad or truncate data."""
    assert truncation in ["left", "right", "error"]

    sequence_length = len(input_ids)
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids, max_length, pad_token_id, left_pad)
        attention_mask = pad_sequence_to_length(attention_mask, max_length, 0, left_pad)
        position_ids = pad_sequence_to_length(position_ids, max_length, 0, left_pad)
    elif sequence_length > max_length:
        if truncation == "left":
            input_ids = input_ids[..., -max_length:]
            attention_mask = attention_mask[..., -max_length:]
            position_ids = position_ids[..., -max_length:]
        elif truncation == "right":
            input_ids = input_ids[..., :max_length]
            attention_mask = attention_mask[..., :max_length]
            position_ids = position_ids[..., :max_length]
        elif truncation == "error":
            raise RuntimeError(f"Input sequence length {sequence_length} is longer than max length {max_length}.")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}.")

    return input_ids, attention_mask, position_ids


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Get the lr scheduler for constant lr."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: Optional[float] = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    init_lr_ratio: Optional[float] = None,
):
    """
    Creates a learning rate schedule that linearly increases the learning rate ratio from `init_lr_ratio`
    to 1.0 over the first `num_warmup_steps`, then applies a cosine decay from 1.0 down to `min_lr_ratio`
    over the remaining training steps.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        init_lr_ratio (:obj:`float`, `optional`, defaults to None):
            The initial lr ratio at the beginning of warmup. If None, defaults to 0.0.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            warmup_progress = float(current_step) / float(max(1, num_warmup_steps))
            if init_lr_ratio is None:
                return warmup_progress
            return init_lr_ratio + (1.0 - init_lr_ratio) * warmup_progress

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value

    return LambdaLR(optimizer, lr_lambda, last_epoch)