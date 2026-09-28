from __future__ import annotations

import logging
import os
import socket
from collections.abc import Sequence
from typing import Any, cast

import ray
from omegaconf import OmegaConf

from .dataset import LoadedDataset

log = logging.getLogger(__name__)

__all__ = [
    "run_ppo",
]


def run_ppo(
    config: Any,
    train_dataset: Sequence[Any],
    val_dataset: Sequence[Any],
) -> None:
    """Launch VERL PPO training with Agent Lightning agent orchestration."""
    from verl.trainer.main_ppo import get_ppo_ray_runtime_env

    assert train_dataset is not None and len(train_dataset) > 0, "train_dataset must be non-empty"
    assert val_dataset is not None and len(val_dataset) > 0, "val_dataset must be non-empty"

    if not ray.is_initialized():
        default_runtime_env = cast(dict[str, Any], get_ppo_ray_runtime_env())
        ray_init_config = OmegaConf.to_container(
            config.ray_kwargs.get("ray_init", OmegaConf.create({})),
            resolve=True,
        )
        ray_init_kwargs: dict[str, Any] = (
            {str(key): value for key, value in ray_init_config.items()}
            if isinstance(ray_init_config, dict)
            else {}
        )
        runtime_env_config = ray_init_kwargs.pop("runtime_env", {})
        runtime_env_kwargs = dict(runtime_env_config) if isinstance(runtime_env_config, dict) else {}
        runtime_env = {**default_runtime_env, **runtime_env_kwargs}
        runtime_env.setdefault(
            "worker_process_setup_hook",
            "agentlightning.verl.per_rollout_loss.register_in_worker",
        )

        temp_dir = os.environ.get("RAY_TMPDIR")
        ray.init(
            runtime_env=runtime_env,
            **({"_temp_dir": temp_dir} if temp_dir else {}),
            **ray_init_kwargs,
        )

    loaded_train_dataset = LoadedDataset(train_dataset)
    loaded_val_dataset = LoadedDataset(val_dataset)

    runner = cast(Any, _AglTaskRunner).remote()
    ray.get(runner.run.remote(config, loaded_train_dataset, loaded_val_dataset))


@ray.remote(num_cpus=1)
class _AglTaskRunner:
    """VERL task runner wrapper accepting already-loaded datasets."""

    def __init__(self):
        from verl.trainer.main_ppo import TaskRunner

        self._delegate = TaskRunner()

    def run(self, config, train_dataset, val_dataset):
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.trainer.main_ppo import (
            create_rl_sampler,
            need_critic,
            need_reference_policy,
            validate_config,
        )
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.utils.tokenizer import hf_processor, hf_tokenizer

        from agentlightning.verl.trainer import AgentLightningRayPPOTrainer

        print(f"AglTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        delegate = self._delegate
        actor_rollout_cls, ray_worker_group_cls = delegate.add_actor_rollout_worker(config)
        delegate.add_critic_worker(config)
        delegate.add_reward_model_resource_pool(config)
        delegate.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if processor is None:
            from transformers import AutoProcessor, PreTrainedTokenizerBase

            try:
                fallback_processor = AutoProcessor.from_pretrained(
                    local_path,
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                )
            except Exception as exc:
                log.warning(
                    "AutoProcessor fallback failed for %s; multimodal training inputs will be disabled: %s",
                    local_path,
                    exc,
                )
                fallback_processor = None

            if fallback_processor is not None and (
                isinstance(fallback_processor, PreTrainedTokenizerBase)
                or "Processor" not in fallback_processor.__class__.__name__
            ):
                fallback_processor = None
            processor = fallback_processor

        resource_pool_manager = delegate.init_resource_pool_mgr(config)

        assert train_dataset is not None and len(train_dataset) > 0, "train_dataset must be non-empty"
        assert val_dataset is not None and len(val_dataset) > 0, "val_dataset must be non-empty"

        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = AgentLightningRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=delegate.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()