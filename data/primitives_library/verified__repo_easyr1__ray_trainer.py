import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for pool_name, processes in self.resource_pool_spec.items():
            self.resource_pool_dict[pool_name] = RayResourcePool(
                process_on_nodes=processes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=pool_name,
            )
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum(gpu_count for node_counts in self.resource_pool_spec.values() for gpu_count in node_counts)

    def _check_resource_available(self):
        available_gpus = ray.available_resources().get("GPU", 0)
        requested_gpus = self.get_num_gpus()
        if available_gpus < requested_gpus:
            raise ValueError(
                f"Total available GPUs {available_gpus} is less than total desired GPUs {requested_gpus}."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    scores = data.batch["token_level_scores"]
    mask = data.batch["response_mask"]
    batch_size = data.batch.batch_size[0]
    kl = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kl = kl * mask
    data.batch["token_level_rewards"] = scores - kl_ctrl.kl_coef * kl
    current_kl = torch.mean(VF.masked_mean(kl, mask=mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
):
    arguments = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        arguments["values"] = data.batch["values"]
    if "reward_baselines" in data.batch:
        arguments["reward_baselines"] = data.batch["reward_baselines"]
    advantages, returns = compute_advantage_return(adv_estimator, **arguments)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None
        self.global_steps = 0
        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.actor_rollout_wg = None
        self.critic_wg = None
        self.ref_policy_wg = None
        self.reward_model_wg = None

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        self.use_critic = config.algorithm.adv_estimator == AdvantageEstimator.GAE
        self._validate_config()

    def _validate_config(self):
        data_config = self.config.data
        actor_config = self.config.worker.actor

        if data_config.rollout_batch_size % actor_config.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        actor_experience_batch_size = actor_config.micro_batch_size_per_device_for_experience
        if (data_config.rollout_batch_size * self.config.worker.rollout.n) % actor_experience_batch_size != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            critic_config = self.config.worker.critic
            if data_config.rollout_batch_size % critic_config.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")
            critic_experience_batch_size = critic_config.micro_batch_size_per_device_for_experience
            if (data_config.rollout_batch_size * self.config.worker.rollout.n) % critic_experience_batch_size != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if self.use_reference_policy and Role.RefPolicy not in self.role_worker_mapping:
            colocated = Role.ActorRolloutRef in self.role_worker_mapping
            if not colocated:
                raise ValueError("Reference policy is enabled but no reference-policy worker is configured.")

        if self.use_critic and Role.Critic not in self.role_worker_mapping:
            raise ValueError("GAE advantage estimation requires a critic worker.")

    def _make_worker_group(self, role: Role, name_prefix: str):
        worker_class = self.role_worker_mapping[role]
        init_args = RayClassWithInitArgs(cls=worker_class, config=self.config.worker, role=name_prefix)
        return self.ray_worker_group_cls(
            resource_pool=self.resource_pool_manager.get_resource_pool(role),
            ray_cls_with_init=init_args,
            name_prefix=name_prefix,
        )

    def _init_worker_group(self):
        self.resource_pool_manager.create_resource_pool()

        if Role.ActorRolloutRef in self.role_worker_mapping:
            worker_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef],
                config=self.config.worker,
                role="actor_rollout_ref",
            )
            self.actor_rollout_wg = self.ray_worker_group_cls(
                resource_pool=self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef),
                ray_cls_with_init=worker_cls,
                name_prefix="actor_rollout_ref",
            )
            self.ref_policy_wg = self.actor_rollout_wg
        elif Role.ActorRollout in self.role_worker_mapping:
            self.actor_rollout_wg = self._make_worker_group(Role.ActorRollout, "actor_rollout")
        else:
            colocated_class = create_colocated_worker_cls(
                class_dict={
                    "actor": self.role_worker_mapping[Role.Actor],
                    "rollout": self.role_worker_mapping[Role.Rollout],
                }
            )
            init_args = RayClassWithInitArgs(cls=colocated_class, config=self.config.worker, role="actor_rollout")
            self.actor_rollout_wg = self.ray_worker_group_cls(
                resource_pool=self.resource_pool_manager.get_resource_pool(Role.Actor),
                ray_cls_with_init=init_args,
                name_prefix="actor_rollout",
            )

        if self.use_critic:
            self.critic_wg = self._make_worker_group(Role.Critic, "critic")

        if self.use_reference_policy and self.ref_policy_wg is None:
            self.ref_policy_wg = self._make_worker_group(Role.RefPolicy, "ref")

        if self.use_reward_model:
            self.reward_model_wg = self._make_worker_group(Role.RewardModel, "reward_model")

    def init_workers(self):
        if self.actor_rollout_wg is None:
            self._init_worker_group()

        groups = []
        for group in (
            self.actor_rollout_wg,
            self.critic_wg,
            self.ref_policy_wg if self.ref_policy_wg is not self.actor_rollout_wg else None,
            self.reward_model_wg,
        ):
            if group is not None:
                groups.append(group)

        for group in groups:
            init_method = getattr(group, "init_model", None)
            if init_method is not None:
                init_method()

    def _ensure_uid(self, batch: DataProto):
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.batch_size[0])])

    def _repeat_data(self, data: DataProto, repeat_times: int):
        if repeat_times == 1:
            return data
        if hasattr(data, "repeat"):
            return data.repeat(repeat_times=repeat_times, interleave=True)
        return data

    def _call_reward_fn(self, reward_fn, data):
        if reward_fn is None:
            return None
        result = reward_fn(data)
        if isinstance(result, DataProto):
            return result
        data.batch["token_level_scores"] = result
        return data

    def _compute_reward(self, data: DataProto):
        if self.use_reward_model and self.reward_model_wg is not None:
            score_fn = getattr(self.reward_model_wg, "compute_rm_score", None)
            if score_fn is not None:
                scores = score_fn(data)
                if isinstance(scores, DataProto):
                    data = scores
                else:
                    data.batch["token_level_scores"] = scores
        elif self.reward_fn is not None:
            result = self._call_reward_fn(self.reward_fn, data)
            if result is not None:
                data = result

        if "token_level_scores" not in data.batch:
            mask = data.batch["response_mask"]
            data.batch["token_level_scores"] = torch.zeros_like(mask, dtype=torch.float32)
        return data

    def _balance_batch(self, data: DataProto):
        if not getattr(self.config.trainer, "balance_batch", False):
            return data
        if "attention_mask" not in data.batch:
            return data
        attention_mask = data.batch["attention_mask"]
        lengths = attention_mask.sum(dim=-1).detach().cpu().tolist()
        world_size = self.resource_pool_manager.get_num_gpus()
        if world_size <= 1:
            return data
        partitions = get_seqlen_balanced_partitions(lengths, world_size, equal_size=True)
        indices = [index for partition in partitions for index in partition]
        log_seqlen_unbalance(lengths, partitions, prefix="global_seqlen")
        if hasattr(data, "reorder"):
            data.reorder(torch.tensor(indices, dtype=torch.long))
        return data

    def _generate_sequences(self, batch: DataProto):
        generator = getattr(self.actor_rollout_wg, "generate_sequences")
        output = generator(batch)
        if isinstance(output, DataProto):
            return output
        return batch

    def _compute_old_log_probs(self, data: DataProto):
        function = getattr(self.actor_rollout_wg, "compute_log_prob", None)
        if function is None:
            return data
        output = function(data)
        if isinstance(output, DataProto):
            if hasattr(data, "union"):
                return data.union(output)
            return output
        if torch.is_tensor(output):
            data.batch["old_log_probs"] = output
        return data

    def _compute_ref_log_probs(self, data: DataProto):
        if not self.use_reference_policy:
            return data
        function = getattr(self.ref_policy_wg, "compute_ref_log_prob", None)
        if function is None:
            function = getattr(self.ref_policy_wg, "compute_log_prob", None)
        if function is None:
            return data
        output = function(data)
        if isinstance(output, DataProto):
            if hasattr(data, "union"):
                return data.union(output)
            return output
        if torch.is_tensor(output):
            data.batch["ref_log_probs"] = output
        return data

    def _compute_values(self, data: DataProto):
        if not self.use_critic:
            return data
        function = getattr(self.critic_wg, "compute_values", None)
        if function is None:
            return data
        output = function(data)
        if isinstance(output, DataProto):
            if hasattr(data, "union"):
                return data.union(output)
            return output
        if torch.is_tensor(output):
            data.batch["values"] = output
        return data

    def _update_critic(self, data: DataProto):
        if not self.use_critic:
            return {}
        update = getattr(self.critic_wg, "update_critic", None)
        if update is None:
            return {}
        result = update(data)
        return result if isinstance(result, dict) else {}

    def _update_actor(self, data: DataProto):
        update = getattr(self.actor_rollout_wg, "update_actor", None)
        if update is None:
            return {}
        result = update(data)
        return result if isinstance(result, dict) else {}

    def _make_data_proto(self, batch):
        if isinstance(batch, DataProto):
            return batch
        if hasattr(DataProto, "from_single_dict"):
            return DataProto.from_single_dict(batch)
        return DataProto(batch=batch, non_tensor_batch={})

    def _step(self, batch: DataProto):
        metrics = {}
        self._ensure_uid(batch)

        with timer("gen", metrics):
            data = self._generate_sequences(batch)

        rollout_n = self.config.worker.rollout.n
        data = self._repeat_data(data, rollout_n)

        with timer("old_log_prob", metrics):
            data = self._compute_old_log_probs(data)

        with timer("ref_log_prob", metrics):
            data = self._compute_ref_log_probs(data)

        with timer("values", metrics):
            data = self._compute_values(data)

        with timer("reward", metrics):
            data = self._compute_reward(data)

        if self.use_reference_policy and "ref_log_probs" in data.batch and "old_log_probs" in data.batch:
            data, kl_metrics = apply_kl_penalty(
                data,
                self.kl_ctrl,
                kl_penalty=getattr(self.config.algorithm, "kl_penalty", "kl"),
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        data = compute_advantage(
            data,
            self.config.algorithm.adv_estimator,
            gamma=getattr(self.config.algorithm, "gamma", 1.0),
            lam=getattr(self.config.algorithm, "lam", 1.0),
        )
        data = self._balance_batch(data)

        with timer("update_critic", metrics):
            metrics.update(self._update_critic(data))

        with timer("update_actor", metrics):
            metrics.update(self._update_actor(data))

        try:
            metrics.update(compute_data_metrics(data, self.use_critic))
        except Exception:
            pass
        try:
            metrics.update(compute_length_metrics(data))
        except Exception:
            pass
        try:
            metrics.update(
                compute_throughout_metrics(
                    data,
                    metrics,
                    self.resource_pool_manager.get_num_gpus(),
                )
            )
        except Exception:
            pass
        try:
            metrics.update(compute_timing_metrics(data, metrics))
        except Exception:
            pass
        return data, metrics

    def validate(self):
        if self.val_dataloader is None:
            return {}

        rewards = []
        for raw_batch in self.val_dataloader:
            batch = self._make_data_proto(raw_batch)
            self._ensure_uid(batch)
            output = self._generate_sequences(batch)
            reward_fn = self.val_reward_fn if self.val_reward_fn is not None else self.reward_fn

            if reward_fn is not None:
                output = self._call_reward_fn(reward_fn, output)
            elif self.use_reward_model:
                output = self._compute_reward(output)

            if "token_level_scores" in output.batch:
                scores = output.batch["token_level_scores"]
                mask = output.batch.get("response_mask", None)
                if mask is not None:
                    scores = VF.masked_mean(scores, mask=mask, dim=-1)
                rewards.extend(scores.detach().float().cpu().reshape(-1).tolist())

        self.val_reward_score = float(np.mean(rewards)) if rewards else 0.0
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_steps
        return {
            "val/reward": self.val_reward_score,
            "val/best_reward": self.best_val_reward_score,
            "val/best_global_step": self.best_global_step,
        }

    def save_checkpoint(self):
        trainer_config = self.config.trainer
        root = trainer_config.default_local_dir
        step = self.global_steps
        local_path = os.path.join(root, f"global_step_{step}")
        os.makedirs(local_path, exist_ok=True)

        groups = {
            "actor": self.actor_rollout_wg,
            "critic": self.critic_wg,
            "ref": self.ref_policy_wg if self.ref_policy_wg is not self.actor_rollout_wg else None,
        }
        for name, group in groups.items():
            if group is None:
                continue
            saver = getattr(group, "save_checkpoint", None)
            if saver is not None:
                saver(
                    local_path=os.path.join(local_path, name),
                    hdfs_path=None,
                    global_step=step,
                    max_ckpt_to_keep=getattr(trainer_config, "max_ckpt_to_keep", None),
                )

        state = {
            "global_steps": step,
            "best_val_reward_score": self.best_val_reward_score,
            "best_global_step": self.best_global_step,
        }
        with open(os.path.join(local_path, "trainer_state.json"), "w", encoding="utf-8") as file:
            json.dump(state, file)

        with open(os.path.join(root, CHECKPOINT_TRACKER), "w", encoding="utf-8") as file:
            file.write(str(step))

        max_to_keep = getattr(trainer_config, "max_ckpt_to_keep", None)
        if max_to_keep is not None:
            try:
                remove_obsolete_ckpt(root, max_to_keep)
            except TypeError:
                remove_obsolete_ckpt(root, global_step=step, max_ckpt_to_keep=max_to_keep)

    def load_checkpoint(self):
        root = self.config.trainer.default_local_dir
        checkpoint_path = find_latest_ckpt(root)
        if checkpoint_path is None:
            return 0

        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as file:
                state = json.load(file)
            self.global_steps = int(state.get("global_steps", 0))
            self.best_val_reward_score = float(state.get("best_val_reward_score", -1.0))
            self.best_global_step = state.get("best_global_step")

        groups = {
            "actor": self.actor_rollout_wg,
            "critic": self.critic_wg,
            "ref": self.ref_policy_wg if self.ref_policy_wg is not self.actor_rollout_wg else None,
        }
        for name, group in groups.items():
            if group is None:
                continue
            loader = getattr(group, "load_checkpoint", None)
            if loader is not None:
                loader(
                    local_path=os.path.join(checkpoint_path, name),
                    hdfs_path=None,
                    del_local_after_load=False,
                )
        return self.global_steps

    def fit(self):
        if self.actor_rollout_wg is None:
            self.init_workers()

        trainer_config = self.config.trainer
        tracker = Tracker(
            project_name=getattr(trainer_config, "project_name", None),
            experiment_name=getattr(trainer_config, "experiment_name", None),
            default_backend=getattr(trainer_config, "logger", "console"),
            config=deepcopy(self.config),
        )

        if getattr(trainer_config, "val_before_train", False):
            tracker.log(self.validate(), step=self.global_steps)

        total_epochs = getattr(trainer_config, "total_epochs", 1)
        total_steps = getattr(trainer_config, "total_training_steps", None)
        save_freq = getattr(trainer_config, "save_freq", -1)
        test_freq = getattr(trainer_config, "test_freq", -1)

        progress_total = total_steps
        progress = tqdm(total=progress_total, desc="Training") if progress_total is not None else None

        for _ in range(total_epochs):
            for raw_batch in self.train_dataloader:
                if total_steps is not None and self.global_steps >= total_steps:
                    break
                batch = self._make_data_proto(raw_batch)
                _, metrics = self._step(batch)
                self.global_steps += 1

                try:
                    metrics = reduce_metrics(metrics)
                except Exception:
                    pass
                tracker.log(metrics, step=self.global_steps)

                if progress is not None:
                    progress.update(1)

                if test_freq > 0 and self.global_steps % test_freq == 0:
                    tracker.log(self.validate(), step=self.global_steps)

                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self.save_checkpoint()

            if total_steps is not None and self.global_steps >= total_steps:
                break

        if progress is not None:
            progress.close()

        if getattr(trainer_config, "save_freq", -1) != 0:
            self.save_checkpoint()

        return {
            "global_steps": self.global_steps,
            "val_reward": self.val_reward_score,
            "best_val_reward": self.best_val_reward_score,
            "best_global_step": self.best_global_step,
        }