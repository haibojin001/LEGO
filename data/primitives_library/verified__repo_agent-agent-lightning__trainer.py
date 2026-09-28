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
    """Return the number of examples represented by a dataloader batch."""
    if not batch:
        return 0
    return len(next(iter(batch.values())))


def _grpo_group_metrics(batch: Any) -> dict[str, int]:
    """Return diagnostics for reward groups used by GRPO."""
    uids = batch.non_tensor_batch.get("uid")
    scores = batch.batch.get("token_level_scores")
    if uids is None or scores is None:
        return {}

    groups: dict[Any, list[float]] = defaultdict(list)
    totals = scores.sum(dim=-1).detach().float().cpu().tolist()
    for uid, total in zip(uids, totals, strict=False):
        groups[uid].append(total)

    return {
        "training/n_groups": len(groups),
        "training/n_zero_adv_groups": sum(
            max(values) == min(values) for values in groups.values() if values
        ),
    }


def _same_reward_uid_indices(batch: DataProto) -> list[int]:
    """Return samples whose complete uid group has a single reward value."""
    if "uid" not in batch.non_tensor_batch:
        return []

    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    groups: dict[Any, list[int]] = {}
    for index, uid in enumerate(batch.non_tensor_batch["uid"]):
        groups.setdefault(uid, []).append(index)

    selected: list[int] = []
    for indices in groups.values():
        values = rewards[indices]
        if torch.allclose(values, values[0].expand_as(values)):
            selected.extend(indices)
    return selected


class AgentLightningRayPPOTrainer(RayPPOTrainer):
    """A VERL PPO trainer whose environment rollouts are served by Agent Lightning."""

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
        agentlightning_config = self.config.agentlightning
        return manager_cls(
            agl_base_url=agentlightning_config.agl_base_url,
            agl_key=agentlightning_config.agl_key,
            model=self.config.actor_rollout_ref.model.path,
            step=self.global_steps,
            train_rollout_n=self.config.actor_rollout_ref.rollout.n,
            rollout_timeout_seconds=agentlightning_config.rollout_timeout_seconds,
            hooks=self._ensure_hooks(),
            local_agent_class=agentlightning_config.local.agent_class,
            local_env_map=agentlightning_config.local.env_map,
            k8s_job_template_path=agentlightning_config.k8s.job_template_path,
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
        inflight = int(payload.get("inflight", 0))

        started = time.perf_counter()
        residual = inflight
        while residual > 0:
            response = client.get("/proxy/state")
            response.raise_for_status()
            residual = int(response.json().get("inflight", 0))
            if residual > 0:
                time.sleep(0.25)

        return {
            "training/async/proxy_inflight_at_pause": inflight,
            "training/async/proxy_drain_seconds": time.perf_counter() - started,
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
    def _timestamp(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return None
        return None

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
            submitted = self._timestamp(
                getattr(
                    rollout,
                    "submitted_at",
                    getattr(rollout, "submitted", None),
                )
            )
            running = self._timestamp(
                getattr(
                    rollout,
                    "running_at",
                    getattr(rollout, "started_at", None),
                )
            )
            completed = self._timestamp(
                getattr(
                    rollout,
                    "completed_at",
                    getattr(
                        rollout,
                        "finished_at",
                        getattr(rollout, "ended_at", None),
                    ),
                )
            )

            if submitted is not None and completed is not None:
                totals.append(max(0.0, completed - submitted))
            if running is None:
                n_missing_running += 1
                continue
            if submitted is not None:
                queue_waits.append(max(0.0, running - submitted))
            if completed is not None:
                run_durations.append(max(0.0, completed - running))

        def aggregate(name: str, values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            array = np.asarray(values, dtype=np.float64)
            return {
                f"training/rollout_lifecycle/{name}_mean_seconds": float(array.mean()),
                f"training/rollout_lifecycle/{name}_min_seconds": float(array.min()),
                f"training/rollout_lifecycle/{name}_max_seconds": float(array.max()),
                f"training/rollout_lifecycle/{name}_p50_seconds": float(
                    np.percentile(array, 50)
                ),
                f"training/rollout_lifecycle/{name}_p90_seconds": float(
                    np.percentile(array, 90)
                ),
            }

        metrics: dict[str, Any] = {
            "training/rollout_lifecycle/n_rollouts": len(completed_rollouts),
            "training/rollout_lifecycle/n_missing_running_at": n_missing_running,
        }
        metrics.update(aggregate("queue_wait", queue_waits))
        metrics.update(aggregate("run", run_durations))
        metrics.update(aggregate("total", totals))
        return metrics

    @staticmethod
    def _await_if_needed(value: Any) -> Any:
        if asyncio.iscoroutine(value):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(value)
        return value

    def _call_rollout_manager(
        self,
        manager: Any,
        batch: Any,
        carry_over_rollouts: list[EnqueuedRollout] | None = None,
    ) -> Any:
        method_names = (
            "rollout",
            "run",
            "generate",
            "generate_rollouts",
            "run_rollouts",
        )
        for name in method_names:
            method = getattr(manager, name, None)
            if method is None:
                continue
            attempts = []
            if carry_over_rollouts is not None:
                attempts.extend(
                    [
                        lambda: method(batch, carry_over_rollouts),
                        lambda: method(
                            batch,
                            carry_over_rollouts=carry_over_rollouts,
                        ),
                        lambda: method(
                            batch,
                            previous_carry_over_rollouts=carry_over_rollouts,
                        ),
                    ]
                )
            attempts.append(lambda: method(batch))
            for attempt in attempts:
                try:
                    return self._await_if_needed(attempt())
                except TypeError:
                    continue
        raise AttributeError(
            f"{type(manager).__name__} has no supported rollout entry point"
        )

    def _adapt_rollouts(
        self,
        completed_rollouts: list[CompletedRollout],
        *,
        is_validation: bool = False,
    ) -> DataProto:
        adapter = RolloutAdapter()
        for name in (
            "adapt",
            "convert",
            "to_data_proto",
            "build_batch",
            "create_batch",
        ):
            method = getattr(adapter, name, None)
            if method is None:
                continue
            try:
                result = method(completed_rollouts, is_validation=is_validation)
            except TypeError:
                result = method(completed_rollouts)
            result = self._await_if_needed(result)
            if result is not None:
                return result
        raise AttributeError("RolloutAdapter has no supported adaptation method")

    def _split_rollout_result(
        self,
        result: Any,
    ) -> tuple[list[CompletedRollout], list[EnqueuedRollout]]:
        if result is None:
            return [], []
        if isinstance(result, tuple):
            if len(result) >= 2:
                return list(result[0] or []), list(result[1] or [])
            return list(result[0] or []), []
        if isinstance(result, dict):
            completed = (
                result.get("completed_rollouts")
                or result.get("completed")
                or result.get("rollouts")
                or []
            )
            carry_over = (
                result.get("carry_over_rollouts")
                or result.get("carry_over")
                or result.get("enqueued_rollouts")
                or []
            )
            return list(completed), list(carry_over)
        return list(result), []

    def _compute_advantages(self, batch: DataProto) -> DataProto:
        estimator = self.config.algorithm.adv_estimator
        estimator_value = getattr(estimator, "value", estimator)

        if estimator_value == "rollout_level":
            return compute_rollout_level_advantage(batch)

        if estimator_value == PER_ROLLOUT_MEAN_LOSS_MODE:
            batch = normalize_advantages_by_rollout(batch)

        response_mask = compute_response_mask(batch)
        return compute_advantage(
            batch,
            adv_estimator=estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            response_mask=response_mask,
        )

    def _prepare_training_batch(self, rollout_batch: DataProto) -> DataProto:
        response_mask = compute_response_mask(rollout_batch)
        rollout_batch.batch["response_mask"] = response_mask

        if getattr(self.config.actor_rollout_ref, "ref", None) is not None:
            if getattr(self.config.actor_rollout_ref.ref, "enable", False):
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(rollout_batch)
                rollout_batch = rollout_batch.union(ref_log_prob)

        if getattr(self.config.actor_rollout_ref, "critic", None) is not None:
            if getattr(self.config.actor_rollout_ref.critic, "enable", False):
                values = self.critic_wg.compute_values(rollout_batch)
                rollout_batch = rollout_batch.union(values)

        if getattr(self.config.algorithm, "use_kl_in_reward", False):
            rollout_batch, _ = apply_kl_penalty(
                rollout_batch,
                kl_ctrl=self.kl_ctrl,
                kl_penalty=self.config.algorithm.kl_penalty,
            )

        return self._compute_advantages(rollout_batch)

    def _update_workers(self, batch: DataProto) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        if getattr(self.config.actor_rollout_ref, "critic", None) is not None and getattr(
            self.config.actor_rollout_ref.critic, "enable", False
        ):
            critic_output = self.critic_wg.update_critic(batch)
            if critic_output:
                metrics.update(reduce_metrics(critic_output))

        actor_output = self.actor_rollout_wg.update_actor(batch)
        if actor_output:
            metrics.update(reduce_metrics(actor_output))

        return metrics

    def _training_rollouts(
        self,
        batch: Any,
    ) -> tuple[DataProto, dict[str, Any]]:
        manager_class: type[AglRolloutManagerBase]
        manager_class = AglAsyncRolloutManager if self.is_async else AglRolloutManager
        manager = self._make_rollout_manager(manager_class)

        old_carry_over = self._carry_over_rollouts
        result = self._call_rollout_manager(
            manager,
            batch,
            old_carry_over if self.is_async else None,
        )
        completed, new_carry_over = self._split_rollout_result(result)

        if self.is_async:
            self._carry_over_rollouts = new_carry_over

        metrics: dict[str, Any] = {}
        metrics.update(self._rollout_lifecycle_metrics(completed))
        if self.is_async:
            metrics.update(
                self._compute_async_rollout_metrics(
                    previous_carry_over_rollouts=old_carry_over,
                    completed_rollouts=completed,
                    new_carry_over_rollouts=new_carry_over,
                )
            )

        return self._adapt_rollouts(completed), metrics

    def fit(self) -> None:
        self.init_workers()
        self._ensure_hooks()
        self._resume_gateway()

        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = getattr(self, "global_steps", 0)
        total_epochs = self.config.trainer.total_epochs
        progress = tqdm(
            total=total_epochs * len(self.train_dataloader),
            initial=self.global_steps,
            desc="Training",
        )

        try:
            for epoch in range(total_epochs):
                self.epoch = epoch
                self._train_dataloader_iter = iter(self.train_dataloader)

                for batch_dict in self._train_dataloader_iter:
                    metrics: dict[str, Any] = {}
                    timing_raw: dict[str, float] = {}
                    with marked_timer("step", timing_raw):
                        with marked_timer("rollout", timing_raw):
                            rollout_batch, rollout_metrics = self._training_rollouts(
                                batch_dict
                            )
                        metrics.update(rollout_metrics)

                        if len(rollout_batch) == 0:
                            log.warning("No completed Agent Lightning rollouts at step %s", self.global_steps)
                            self.global_steps += 1
                            progress.update(1)
                            continue

                        with marked_timer("advantage", timing_raw):
                            rollout_batch = self._prepare_training_batch(rollout_batch)

                        metrics.update(_grpo_group_metrics(rollout_batch))
                        metrics.update(compute_data_metrics(rollout_batch))

                        with marked_timer("update_actor", timing_raw):
                            metrics.update(self._update_workers(rollout_batch))

                    metrics.update(compute_timing_metrics(timing_raw))
                    metrics.update(
                        compute_throughout_metrics(
                            batch=rollout_batch,
                            timing_raw=timing_raw,
                            n_gpus=getattr(self, "world_size", 1),
                        )
                    )
                    metrics["training/global_step"] = self.global_steps
                    metrics["training/epoch"] = epoch
                    tracking.log(data=metrics, step=self.global_steps)

                    self.global_steps += 1
                    progress.update(1)

                    if (
                        getattr(self.config.trainer, "save_freq", 0) > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        self._save_checkpoint()

        finally:
            progress.close()
            try:
                self._pause_and_drain_gateway(reason="trainer_shutdown")
            except Exception:
                log.exception("Failed to pause Agent Lightning gateway during shutdown")
            if self._hooks is not None:
                shutdown = getattr(self._hooks, "on_shutdown", None)
                if shutdown is not None:
                    shutdown()