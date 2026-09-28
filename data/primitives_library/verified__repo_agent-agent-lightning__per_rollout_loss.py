from __future__ import annotations

from typing import Any

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss

PER_ROLLOUT_MEAN_LOSS_MODE = "per_rollout_mean"


def normalize_advantages_by_rollout(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_ids: Any,
    *,
    num_trained_rows: int,
) -> torch.Tensor:
    """Normalize each row by its rollout's token count and batch size."""
    if len(rollout_ids) != advantages.shape[0]:
        raise ValueError(
            f"rollout_ids length ({len(rollout_ids)}) must match advantages rows ({advantages.shape[0]})"
        )
    if num_trained_rows <= 0:
        raise ValueError("num_trained_rows must be positive")

    token_counts = response_mask.sum(dim=-1).to(dtype=advantages.dtype)
    totals: dict[Any, float] = {}

    for index, identifier in enumerate(rollout_ids):
        totals[identifier] = totals.get(identifier, 0.0) + float(
            token_counts[index].item()
        )

    denominators = torch.tensor(
        [totals[identifier] * num_trained_rows for identifier in rollout_ids],
        dtype=advantages.dtype,
        device=advantages.device,
    ).clamp_min(1.0)

    return advantages / denominators.unsqueeze(-1)


@register_policy_loss(PER_ROLLOUT_MEAN_LOSS_MODE)
def compute_policy_loss_per_rollout_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute clipped PPO loss from rollout-normalized advantages."""
    assert config is not None, "per_rollout_mean loss requires the actor config"

    clip_ratio = config.clip_ratio
    low_clip_ratio = (
        config.clip_ratio_low
        if config.clip_ratio_low is not None
        else clip_ratio
    )
    high_clip_ratio = (
        config.clip_ratio_high
        if config.clip_ratio_high is not None
        else clip_ratio
    )
    lower_bound_ratio = config.get("clip_ratio_c", 3.0)

    assert lower_bound_ratio > 1.0, (
        f"clip_ratio_c must be greater than 1.0, got {lower_bound_ratio}"
    )

    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

    unclipped_losses = -advantages * ratio
    clipped_losses = -advantages * torch.clamp(
        ratio,
        min=1 - low_clip_ratio,
        max=1 + high_clip_ratio,
    )
    primary_losses = torch.maximum(unclipped_losses, clipped_losses)
    clip_fraction = verl_F.masked_mean(
        torch.gt(clipped_losses, unclipped_losses).float(),
        response_mask,
    )

    lower_losses = -advantages * lower_bound_ratio
    clipped_primary_losses = torch.min(lower_losses, primary_losses)
    lower_clip_fraction = verl_F.masked_mean(
        torch.gt(primary_losses, lower_losses) * (advantages < 0).float(),
        response_mask,
    )
    policy_losses = torch.where(
        advantages < 0,
        clipped_primary_losses,
        primary_losses,
    )

    if rollout_is_weights is not None:
        policy_losses = policy_losses * rollout_is_weights

    dp_size = config.global_batch_info.get("dp_size", 1) if config.global_batch_info else 1
    policy_loss = verl_F.masked_sum(policy_losses, response_mask) * (dp_size or 1)

    return policy_loss, {
        "actor/pg_clipfrac": clip_fraction.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": lower_clip_fraction.detach().item(),
    }


def register_in_worker() -> None:
    """Import hook used by Ray actor processes."""