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

__all__ = ["run_ppo"]


def _ray_init_options(config: Any, default_runtime_env: dict[str, Any]) -> dict[str, Any]:
    configured = config.ray_kwargs.get("ray_init", OmegaConf.create({}))
    resolved = OmegaConf.to_container(configured, resolve=True)
    options = (
        {str(key): value for key, value in resolved.items()}
        if isinstance(resolved, dict)
        else {}
    )

    supplied_runtime_env = options.pop("runtime_env", {})
    supplied_runtime_env = (
        dict(supplied_runtime_env)
        if isinstance(supplied_runtime_env, dict)
        else {}
    )
    runtime_env = {**default_runtime_env, **supplied_runtime_env}
    runtime_env.setdefault(
        "worker_process_setup_hook",
        "agentlightning.verl.per_rollout_loss.register_in_worker",
    )
    options["runtime_env"] = runtime_env

    temp_directory = os.environ.get("RAY_TMPDIR")
    if temp_directory:
        options["_temp_dir"] = temp_directory

    return options


def run_ppo(
    config: Any,
    train_dataset: Sequence[Any],
    val_dataset: Sequence[Any],
) -> None:
    """Launch VERL PPO training using Agent Lightning-managed rollouts."""
    assert train_dataset is not None and len(train_dataset) > 0, (
        "train_dataset must be non-empty"
    )
    assert val_dataset is not None and len(val_dataset) > 0, (
        "val_dataset must be non-empty"
    )

    if not ray.is_initialized():
        from verl.trainer.main_ppo import get_ppo_ray_runtime_env

        baseline_env = cast(dict[str, Any], get_ppo_ray_runtime_env())
        ray.init(**_ray_init_options(config, baseline_env))

    training_data = LoadedDataset(train_dataset)
    validation_data = LoadedDataset(val_dataset)

    task_runner = cast(Any, _AglTaskRunner).remote()
    ray.get(task_runner.run.remote(config, training_data, validation_data))


@ray.remote(num_cpus=1)
class _AglTaskRunner:
    """Ray task actor that prepares VERL workers and runs the custom trainer."""

    def __init__(self) -> None:
        from verl.trainer.main_ppo import TaskRunner

        self._task_runner = TaskRunner()

    def run(self, config: Any, train_dataset: Any, val_dataset: Any) -> None:
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

        verl_runner = self._task_runner
        actor_rollout_cls, worker_group_cls = verl_runner.add_actor_rollout_worker(config)
        verl_runner.add_critic_worker(config)
        verl_runner.add_reward_model_resource_pool(config)
        verl_runner.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        model_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if processor is None:
            from transformers import AutoProcessor, PreTrainedTokenizerBase

            try:
                candidate = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                )
            except Exception as error:
                log.warning(
                    "AutoProcessor fallback failed for %s; multimodal training inputs "
                    "will be disabled: %s",
                    model_path,
                    error,
                )
                candidate = None

            if candidate is not None and (
                isinstance(candidate, PreTrainedTokenizerBase)
                or "Processor" not in candidate.__class__.__name__
            ):
                candidate = None

            processor = candidate

        pool_manager = verl_runner.init_resource_pool_mgr(config)

        assert train_dataset is not None and len(train_dataset) > 0, (
            "train_dataset must be non-empty"
        )
        assert val_dataset is not None and len(val_dataset) > 0, (
            "val_dataset must be non-empty"
        )

        sampler = create_rl_sampler(config.data, train_dataset)

        trainer = AgentLightningRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=verl_runner.role_worker_mapping,
            resource_pool_manager=pool_manager,
            ray_worker_group_cls=worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=sampler,
        )
        trainer.init_workers()
        trainer.fit()