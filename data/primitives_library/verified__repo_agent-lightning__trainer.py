from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from typing import Any, TypeVar

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import marked_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import Tracking

from agentlightning.client import AgentLightningSyncClient
from agentlightning.hooks import RolloutHooks, load_hooks

from .agl_rollout_manager import (
    AglAsyncRolloutManager,
    AglRolloutManager,
    AglRolloutManagerBase,
    CompletedRollout,
    EnqueuedRollout,
)
from .per_rollout_loss import PER_ROLLOUT_MEAN_LOSS_MODE, normalize_advantages_by_rollout
from .rollout_adapter import RolloutAdapter
from .rollout_level_advantage import compute_rollout_level_advantage

log = logging.getLogger(__name__)

RolloutManagerT = TypeVar("RolloutManagerT", bound=AglRolloutManagerBase)


def _batch_dict_len(batch: dict[str, Any] | None) -> int:
    if not batch:
        return 0
    return len(next(iter(batch.values())))


def _grpo_group_metrics(batch: Any) -> dict[str, int]:
    uids = batch.non_tensor_batch.get("uid")
    scores = batch.batch.get("token_level_scores")
    if uids is None or scores is None:
        return {}

    grouped_scores: dict[Any, list[float]] = defaultdict(list)
    values = scores.sum(dim=-1).detach().float().cpu().tolist()
    for uid, score in zip(uids, values, strict=False):
        grouped_scores[uid].append(score)

    zero_variance_groups = sum(
        max(group) == min(group) for group in grouped_scores.values() if group
    )
    return {
        "training/n_groups": len(grouped_scores),
        "training/n_zero_adv_groups": zero_variance_groups,
    }


def _same_reward_uid_indices(batch: DataProto) -> list[int]:
    if "uid" not in batch.non_tensor_batch:
        return []

    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    grouped_indices: dict[Any, list[int]] = {}
    for index, uid in enumerate(batch.non_tensor_batch["uid"]):
        grouped_indices.setdefault(uid, []).append(index)

    result: list[int] = []
    for indices in grouped_indices.values():
        group_rewards = rewards[indices]
        if torch.allclose(group_rewards, group_rewards[0].expand_as(group_rewards)):
            result.extend(indices)
    return result


class AgentLightningRayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.is_async = self.config.agentlightning.async_rollout.enabled
        self.epoch = 0

        async_train_batch_size = self.config.agentlightning.async_rollout.async_train_batch_size
        train_batch_size = self.config.data.train_batch_size
        if self.is_async and async_train_batch_size <= train_batch_size:
            raise ValueError(
                f"async_train_batch_size ({async_train_batch_size}) must be > "
                f"data.train_batch_size ({train_batch_size})."
            )

        self._hooks: RolloutHooks | None = None
        self._agl_client: AgentLightningSyncClient | None = None
        self._carry_over_rollouts: list[EnqueuedRollout] = []
        self._train_dataloader_iter: Any | None = None

    def _ensure_hooks(self) -> RolloutHooks | None:
        if self._hooks is not None:
            return self._hooks

        hooks_path = self.config.agentlightning.hooks
        if not hooks_path:
            return None

        self._hooks = load_hooks(hooks_path)
        self._hooks.on_startup()
        return self._hooks

    def _ensure_agl_client(self) -> AgentLightningSyncClient:
        if self._agl_client is None:
            self._agl_client = AgentLightningSyncClient(
                base_url=self.config.agentlightning.agl_base_url,
                key=self.config.agentlightning.agl_key,
                timeout=300,
            )
        return self._agl_client

    def _make_rollout_manager(self, manager_cls: type[RolloutManagerT]) -> RolloutManagerT:
        config = self.config.agentlightning
        return manager_cls(
            agl_base_url=config.agl_base_url,
            agl_key=config.agl_key,
            model=self.config.actor_rollout_ref.model.path,
            step=self.global_steps,
            train_rollout_n=self.config.actor_rollout_ref.rollout.n,
            rollout_timeout_seconds=config.rollout_timeout_seconds,
            hooks=self._ensure_hooks(),
            local_agent_class=config.local.agent_class,
            local_env_map=config.local.env_map,
            k8s_job_template_path=config.k8s.job_template_path,
        )

    def _rollout_replicas(self) -> list[Any]:
        if hasattr(self, "llm_server_manager"):
            return list(self.llm_server_manager.get_replicas())
        return list(self.async_rollout_manager.rollout_replicas)

    @auto_await
    async def _abort_all_rollout_requests(self) -> None:
        await asyncio.gather(
            *(replica.abort_all_requests() for replica in self._rollout_replicas())
        )

    @auto_await
    async def _resume_all_rollout_generation(self) -> None:
        await asyncio.gather(
            *(replica.resume_generation() for replica in self._rollout_replicas())
        )

    def _resume_gateway(self) -> None:
        self._ensure_agl_client().post_with_retry("/proxy/resume")

    def _pause_and_drain_gateway(self, *, reason: str) -> dict[str, Any]:
        client = self._ensure_agl_client()
        response = client.post_with_retry(
            "/proxy/pause",
            json={"reason": reason},
            timeout=300.0,
        )
        payload = response.json()
        inflight_at_pause = int(payload.get("inflight", 0))

        started_at = time.perf_counter()
        inflight = inflight_at_pause
        while True:
            response = client.get("/proxy/state")
            response.raise_for_status()
            inflight = int(response.json().get("inflight", 0))
            if inflight <= 0:
                break
            time.sleep(0.25)

        return {
            "training/async/proxy_inflight_at_pause": inflight_at_pause,
            "training/async/proxy_drain_seconds": time.perf_counter() - started_at,
        }

    def _compute_async_rollout_metrics(
        self,
        *,
        previous_carry_over_rollouts: list[EnqueuedRollout],
        completed_rollouts: list[CompletedRollout],
        new_carry_over_rollouts: list[EnqueuedRollout],
    ) -> dict[str, Any]:
        max_age = max(
            (
                self.global_steps - rollout.step
                for rollout in new_carry_over_rollouts
            ),
            default=0,
        )
        return {
            "training/async/n_prev_carry_over_rollouts": len(
                previous_carry_over_rollouts
            ),
            "training/async/n_completed_rollouts": len(completed_rollouts),
            "training/async/n_new_carry_over_rollouts": len(
                new_carry_over_rollouts
            ),
            "training/async/new_carry_over_age_max_steps": max_age,
        }

    @staticmethod
    def _timestamp_seconds(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return None
        if hasattr(value, "timestamp"):
            try:
                return float(value.timestamp())
            except Exception:
                return None
        return None

    @staticmethod
    def _rollout_field(rollout: Any, *names: str) -> Any:
        for name in names:
            if isinstance(rollout, dict) and name in rollout:
                return rollout[name]
            if hasattr(rollout, name):
                return getattr(rollout, name)
        nested = getattr(rollout, "rollout", None)
        if nested is not None and nested is not rollout:
            for name in names:
                if isinstance(nested, dict) and name in nested:
                    return nested[name]
                if hasattr(nested, name):
                    return getattr(nested, name)
        return None

    @staticmethod
    def _timing_summary(prefix: str, values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}/mean": float(array.mean()),
            f"{prefix}/min": float(array.min()),
            f"{prefix}/max": float(array.max()),
            f"{prefix}/p50": float(np.percentile(array, 50)),
            f"{prefix}/p90": float(np.percentile(array, 90)),
        }

    def _rollout_lifecycle_metrics(
        self,
        completed_rollouts: list[CompletedRollout],
    ) -> dict[str, Any]:
        if not completed_rollouts:
            return {}

        queue_waits: list[float] = []
        run_durations: list[float] = []
        totals: list[float] = []
        n_missing_running = 0

        for rollout in completed_rollouts:
            submitted = self._timestamp_seconds(
                self._rollout_field(
                    rollout,
                    "submitted_at",
                    "submitted_time",
                    "created_at",
                    "enqueued_at",
                )
            )
            running = self._timestamp_seconds(
                self._rollout_field(
                    rollout,
                    "running_at",
                    "started_at",
                    "started_time",
                )
            )
            completed = self._timestamp_seconds(
                self._rollout_field(
                    rollout,
                    "completed_at",
                    "finished_at",
                    "completed_time",
                    "ended_at",
                )
            )

            if running is None:
                n_missing_running += 1
            elif submitted is not None:
                queue_waits.append(max(0.0, running - submitted))

            if completed is not None and running is not None:
                run_durations.append(max(0.0, completed - running))

            if completed is not None and submitted is not None:
                totals.append(max(0.0, completed - submitted))

        metrics: dict[str, Any] = {
            "training/rollout_lifecycle/n_completed_rollouts": len(completed_rollouts),
            "training/rollout_lifecycle/n_missing_running_at": n_missing_running,
        }
        metrics.update(
            self._timing_summary(
                "training/rollout_lifecycle/queue_wait_seconds",
                queue_waits,
            )
        )
        metrics.update(
            self._timing_summary(
                "training/rollout_lifecycle/run_seconds",
                run_durations,
            )
        )
        metrics.update(
            self._timing_summary(
                "training/rollout_lifecycle/total_seconds",
                totals,
            )
        )
        return metrics

    def _next_train_batch(self) -> dict[str, Any]:
        if self._train_dataloader_iter is None:
            self._train_dataloader_iter = iter(self.train_dataloader)
        try:
            return next(self._train_dataloader_iter)
        except StopIteration:
            self.epoch += 1
            self._train_dataloader_iter = iter(self.train_dataloader)
            return next(self._train_dataloader_iter)

    def _new_rollout_id(self) -> str:
        return str(uuid.uuid4())

    def _seed_for_step(self) -> None:
        seed = getattr(self.config, "seed", None)
        if seed is None:
            return
        value = int(seed) + int(self.global_steps)
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def _apply_rollout_advantage(
        self,
        batch: DataProto,
    ) -> DataProto:
        estimator = self.config.algorithm.adv_estimator
        if estimator == AdvantageEstimator.GRPO:
            return compute_rollout_level_advantage(batch)
        return batch

    def _normalize_rollout_advantages(self, batch: DataProto) -> DataProto:
        loss_agg_mode = getattr(
            self.config.actor_rollout_ref.actor,
            "loss_agg_mode",
            None,
        )
        if loss_agg_mode == PER_ROLLOUT_MEAN_LOSS_MODE:
            return normalize_advantages_by_rollout(batch)
        return batch

    def _close_agentlightning_resources(self) -> None:
        hooks = self._hooks
        if hooks is not None:
            shutdown = getattr(hooks, "on_shutdown", None)
            if callable(shutdown):
                shutdown()
        self._hooks = None
        self._agl_client = None

    def __del__(self) -> None:
        try:
            self._close_agentlightning_resources()
        except Exception:
            pass