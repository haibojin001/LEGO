from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF

if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        ...


class AdaptiveKLController(KLController):
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        self.kl_coef *= 1 + proportional_error * n_steps / self.horizon


class FixedKLController(KLController):
    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")
    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]

    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        group_scores = torch.tensor(id2score[idx])
        id2mean[idx] = torch.mean(group_scores)
        id2std[idx] = torch.std(group_scores)

    for i in range(scores.shape[0]):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        group_scores = torch.tensor(id2score[idx])
        top_values, top_indices = torch.topk(group_scores, k=2)
        winner = id2indices[idx][top_indices[0]]
        advantages[winner] = (top_values[0] - top_values[1]) / (torch.std(group_scores) + eps)

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)

    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])

    advantages = torch.zeros_like(scores)
    for i in range(scores.shape[0]):
        group_scores = id2score[index[i]]
        assert len(group_scores) > 1, "RLOO needs rollout.n > 1."
        baseline = (sum(group_scores) - scores[i]) / (len(group_scores) - 1)
        advantages[i] = scores[i] - baseline

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0

    for t in reversed(range(token_level_rewards.shape[-1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    reward_baselines: torch.Tensor,
    gamma: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0

    for t in reversed(range(token_level_rewards.shape[-1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    advantages = returns - reward_baselines.unsqueeze(-1)
    advantages = advantages * response_mask
    return advantages, returns


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: Literal[
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    ] = "token-mean",
) -> torch.Tensor:
    if loss_agg_mode == "token-mean":
        return VF.masked_mean(loss_mat, loss_mask)
    if loss_agg_mode == "seq-mean-token-sum":
        return torch.mean(torch.sum(loss_mat * loss_mask, dim=-1))
    if loss_agg_mode == "seq-mean-token-mean":
        return torch.mean(torch.sum(loss_mat * loss_mask, dim=-1) / loss_mask.sum(dim=-1))
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return torch.mean(torch.sum(loss_mat * loss_mask, dim=-1) / torch.sqrt(loss_mask.sum(dim=-1)))
    raise ValueError(f"Unknown loss aggregation mode: {loss_agg_mode}")


def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(log_prob - old_log_prob)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = agg_loss(torch.maximum(pg_losses, pg_losses2), response_mask, loss_agg_mode)
    pg_clipfrac = VF.masked_mean((pg_losses2 > pg_losses).to(torch.float32), response_mask)
    return pg_loss, pg_clipfrac


def compute_entropy_loss(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    entropy = VF.entropy_from_logits(logits)
    return agg_loss(entropy, response_mask, loss_agg_mode)


def compute_value_loss(
    vpreds: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    vpred_clipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpred_clipped - returns) ** 2
    vf_loss = 0.5 * agg_loss(torch.maximum(vf_losses1, vf_losses2), response_mask, loss_agg_mode)
    vf_clipfrac = VF.masked_mean((vf_losses2 > vf_losses1).to(torch.float32), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    kl_penalty: str,
) -> torch.Tensor:
    log_ratio = logprob - ref_logprob

    if kl_penalty == "kl":
        return log_ratio
    if kl_penalty == "abs":
        return log_ratio.abs()
    if kl_penalty == "mse":
        return 0.5 * log_ratio.square()
    if kl_penalty == "low_var_kl":
        return torch.exp(-log_ratio) + log_ratio - 1.0

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}")