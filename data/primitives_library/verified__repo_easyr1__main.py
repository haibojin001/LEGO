import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class Runner:
    """Coordinates distributed RL training setup and execution."""

    def run(self, config: PPOConfig):
        print(json.dumps(config.to_dict(), indent=2))

        actor_model = config.worker.actor.model
        tokenizer = get_tokenizer(
            actor_model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=actor_model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            actor_model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=actor_model.trust_remote_code,
            use_fast=True,
        )

        pool_id = "global_pool"
        pool_specification = {
            pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        role_pool_mapping = {
            Role.ActorRolloutRef: pool_id,
            Role.Critic: pool_id,
        }
        pools = ResourcePoolManager(
            resource_pool_spec=pool_specification,
            mapping=role_pool_mapping,
        )

        workers = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }

        reward_actor = ray.remote(AutoRewardManager).options(
            num_cpus=config.worker.reward.num_cpus
        )
        reward_fn = reward_actor.remote(config.worker.reward, tokenizer)
        validation_reward_fn = reward_actor.remote(config.worker.reward, tokenizer)

        train_loader, validation_loader = create_dataloader(
            config.data,
            tokenizer,
            processor,
        )

        training_engine = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_loader,
            val_dataloader=validation_loader,
            role_worker_mapping=workers,
            resource_pool_manager=pools,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=validation_reward_fn,
        )
        training_engine.init_workers()
        training_engine.fit()


def main():
    cli_config = OmegaConf.from_cli()
    config_schema = OmegaConf.structured(PPOConfig())

    if hasattr(cli_config, "config"):
        config_filename = cli_config.pop("config", None)
        config_schema = OmegaConf.merge(
            config_schema,
            OmegaConf.load(config_filename),
        )

    resolved_config = OmegaConf.merge(config_schema, cli_config)
    resolved_config: PPOConfig = OmegaConf.to_object(resolved_config)
    resolved_config.deep_post_init()

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                }
            }
        )

    task = Runner.remote()
    ray.get(task.run.remote(resolved_config))

    if resolved_config.trainer.ray_timeline is not None:
        ray.timeline(filename=resolved_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()