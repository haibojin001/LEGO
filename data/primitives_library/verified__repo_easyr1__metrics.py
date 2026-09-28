from typing import Any

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {metric_name: np.mean(samples) for metric_name, samples in metrics.items()}


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    response_size = batch.batch["responses"].size(-1)
    prompt_size = batch.batch["attention_mask"].size(-1) - response_size
    attention = batch.batch["attention_mask"]

    prompt_counts = attention[:, :-response_size].sum(dim=-1).float()
    response_counts = attention[:, -response_size:].sum(dim=-1).float()

    return {
        "response_length/mean": response_counts.mean().detach().item(),
        "response_length/max": response_counts.max().detach().item(),
        "response_length/min": response_counts.min().detach().item(),
        "response_length/clip_ratio": response_counts.eq(response_size).float().mean().detach().item(),
        "prompt_length/mean": prompt_counts.mean().detach().item(),
        "prompt_length/max": prompt_counts.max().detach().item(),
        "prompt_length/min": prompt_counts.min().detach().item(),
        "prompt_length/clip_ratio": prompt_counts.eq(prompt_size).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    scores = batch.batch["token_level_scores"].sum(dim=-1)
    rewards = batch.batch["token_level_rewards"].sum(dim=-1)

    response_size = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -response_size:].bool()

    selected_advantages = torch.masked_select(batch.batch["advantages"], response_mask)
    selected_returns = torch.masked_select(batch.batch["returns"], response_mask)

    critic_metrics: dict[str, Any] = {}
    if use_critic:
        selected_values = torch.masked_select(batch.batch["values"], response_mask)
        residual_variance = torch.var(selected_returns - selected_values)
        return_variance = torch.var(selected_returns)
        critic_metrics = {
            "critic/values/mean": selected_values.mean().detach().item(),
            "critic/values/max": selected_values.max().detach().item(),
            "critic/values/min": selected_values.min().detach().item(),
            "critic/vf_explained_var": (
                1.0 - residual_variance / (return_variance + 1e-5)
            ).detach().item(),
        }

    metrics = {
        "critic/score/mean": scores.mean().detach().item(),
        "critic/score/max": scores.max().detach().item(),
        "critic/score/min": scores.min().detach().item(),
        "critic/rewards/mean": rewards.mean().detach().item(),
        "critic/rewards/max": rewards.max().detach().item(),
        "critic/rewards/min": rewards.min().detach().item(),
        "critic/advantages/mean": selected_advantages.mean().detach().item(),
        "critic/advantages/max": selected_advantages.max().detach().item(),
        "critic/advantages/min": selected_advantages.min().detach().item(),
        "critic/returns/mean": selected_returns.mean().detach().item(),
        "critic/returns/max": selected_returns.max().detach().item(),
        "critic/returns/min": selected_returns.min().detach().item(),
    }
    metrics.update(critic_metrics)
    metrics.update(compute_length_metrics(batch))
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    response_token_count = torch.sum(batch.batch["response_mask"]).item()
    overall_token_count = sum(batch.meta_info["global_token_num"])

    token_counts = {
        **dict.fromkeys(("gen", "reward"), response_token_count),
        **dict.fromkeys(
            ("ref", "old", "values", "adv", "update_critic", "update_actor"),
            overall_token_count,
        ),
    }

    result = {f"timing_s/{section}": elapsed for section, elapsed in timing_raw.items()}
    result.update(
        {
            f"timing_per_token_ms/{section}": timing_raw[section] * 1000 / token_counts[section]
            for section in set(token_counts).intersection(timing_raw)
        }
    )
    return result


def compute_throughout_metrics(
    batch: DataProto, timing_raw: dict[str, float], num_gpus: int
) -> dict[str, Any]:
    token_count = sum(batch.meta_info["global_token_num"])
    step_time = timing_raw["step"]

    return {
        "perf/total_num_tokens": token_count,
        "perf/time_per_step": step_time,
        "perf/throughput": token_count / (step_time * num_gpus),
    }