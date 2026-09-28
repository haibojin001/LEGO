from contextlib import nullcontext
from typing import Literal, Optional, Union, cast

import numpy as np
import peft
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from peft import TaskType, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


def _value(obj, name, default=None):
    return getattr(obj, name, default)


def _precision(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    return PrecisionType.to_dtype(value)


def _dispatch(mode):
    try:
        return register(dispatch_mode=mode)
    except TypeError:
        return register(mode)


def _make_proto(tensors, meta_info=None):
    meta_info = {} if meta_info is None else meta_info
    try:
        return DataProto.from_dict(tensors=tensors, meta_info=meta_info)
    except TypeError:
        try:
            return DataProto(batch=tensors, meta_info=meta_info)
        except TypeError:
            return DataProto(tensors, meta_info=meta_info)


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = role in ("actor", "actor_rollout", "actor_rollout_ref")
        self._has_critic = role == "critic"
        self._has_rollout = role in ("rollout", "actor_rollout", "actor_rollout_ref")
        self._has_ref = role in ("ref", "actor_rollout_ref")

        if self._has_actor and self._has_critic:
            raise ValueError("actor and critic cannot coexist in one FSDP worker")

        if _value(_value(config, "actor"), "disable_kl", False):
            self._has_ref = False

        self._lora_rank = _value(_value(_value(config, "actor"), "model"), "lora").rank
        self._is_lora = self._lora_rank > 0
        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False

        if self._has_actor:
            self._use_param_offload = _value(config.actor.offload, "offload_params", False)
            self._use_optimizer_offload = _value(config.actor.offload, "offload_optimizer", False)
            self._init_dist_mesh(config.actor, "actor")
        elif self._has_critic:
            self._use_param_offload = _value(config.critic.offload, "offload_params", False)
            self._use_optimizer_offload = _value(config.critic.offload, "offload_optimizer", False)
            self._init_dist_mesh(config.critic, "critic")

        if self._has_ref:
            self._use_ref_param_offload = _value(config.ref.offload, "offload_params", False)

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        fsdp_size = _value(config.fsdp, "fsdp_size", -1)

        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        else:
            if world_size % fsdp_size:
                raise ValueError("world size must be divisible by fsdp_size")
            self.device_mesh = init_device_mesh(
                "cuda",
                (world_size // fsdp_size, fsdp_size),
                mesh_dim_names=("ddp", "fsdp"),
            )

        ulysses_size = _value(config, "ulysses_size", 1)
        if ulysses_size > 1:
            if world_size % ulysses_size:
                raise ValueError("world size must be divisible by ulysses_size")
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                (world_size // ulysses_size, ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        if _value(self.config.rollout, "n", 1) > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} global batch size: {config.global_batch_size}")

        denominator = world_size // ulysses_size
        config.global_batch_size_per_device = config.global_batch_size // denominator

        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size is too small for the configured devices")

        micro = config.micro_batch_size_per_device_for_update
        if config.global_batch_size_per_device % micro:
            raise ValueError(f"{role} global batch size per device must divide exactly into update micro batches")

        if _value(config.fsdp, "enable_cpu_offload", False) and config.global_batch_size_per_device != micro:
            raise ValueError("FSDP CPU offload cannot be combined with gradient accumulation")

    def _fsdp_kwargs(self, fsdp_config):
        dtype = _precision(_value(fsdp_config, "torch_dtype"))
        mixed = None
        if dtype is not None and dtype != torch.float32:
            mixed = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

        strategy_name = str(_value(fsdp_config, "sharding_strategy", "full_shard")).lower()
        strategies = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
            "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        }
        strategy = strategies.get(strategy_name, ShardingStrategy.FULL_SHARD)
        cpu_offload = CPUOffload(offload_params=True) if _value(fsdp_config, "enable_cpu_offload", False) else None

        kwargs = {
            "device_mesh": self.device_mesh,
            "sharding_strategy": strategy,
            "mixed_precision": mixed,
            "cpu_offload": cpu_offload,
            "use_orig_params": _value(fsdp_config, "use_orig_params", True),
            "sync_module_states": _value(fsdp_config, "enable_rank0_init", False),
            "limit_all_gathers": _value(fsdp_config, "limit_all_gathers", True),
        }
        policy = get_fsdp_wrap_policy(
            model_config=self.model_config,
            fsdp_config=fsdp_config,
        )
        if policy is not None:
            kwargs["auto_wrap_policy"] = policy
        return kwargs

    def _create_optimizer(self, model, config):
        if config is None:
            return None

        params = [p for p in model.parameters() if p.requires_grad]
        name = str(_value(config, "optimizer", _value(config, "name", "adamw"))).lower()
        kwargs = {
            "lr": _value(config, "lr", 1e-5),
            "weight_decay": _value(config, "weight_decay", 0.0),
        }

        if name in ("adamw", "torch_adamw"):
            optimizer = torch.optim.AdamW(
                params,
                betas=tuple(_value(config, "betas", (_value(config, "beta1", 0.9), _value(config, "beta2", 0.999)))),
                eps=_value(config, "eps", 1e-8),
                **kwargs,
            )
        else:
            optimizer = AnyPrecisionAdamW(
                params,
                betas=tuple(_value(config, "betas", (0.9, 0.999))),
                eps=_value(config, "eps", 1e-8),
                **kwargs,
            )

        total_steps = _value(config, "total_training_steps", _value(config, "total_steps", 0))
        warmup = _value(config, "warmup_steps", 0)
        scheduler_type = str(_value(config, "lr_scheduler", _value(config, "scheduler", "constant"))).lower()
        if total_steps:
            if scheduler_type == "cosine":
                scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
            else:
                scheduler = get_constant_schedule_with_warmup(optimizer, warmup)
        else:
            scheduler = None
        return optimizer, scheduler

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        if role != "ref":
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **_value(model_config, "override_config", {}),
            )
            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)
            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied")

        configured_dtype = _value(fsdp_config, "torch_dtype")
        torch_dtype = _precision(configured_dtype)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if role == "ref" else torch.float32

        if role == "critic":
            model_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForCausalLM

        rank_zero = self.device_mesh.get_local_rank("fsdp") == 0
        rank0_init = _value(fsdp_config, "enable_rank0_init", False)
        construct_real = not rank0_init or rank_zero

        if construct_real:
            model = model_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if rank0_init else None,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with init_empty_weights(), no_init_weights():
                model = model_class.from_config(
                    self.model_config,
                    trust_remote_code=model_config.trust_remote_code,
                )

        if role == "critic" and _value(self.model_config, "num_labels", None) != 1:
            self.model_config.num_labels = 1

        lora = _value(model_config, "lora")
        if role != "ref" and _value(lora, "rank", 0) > 0:
            targets = _value(lora, "target_modules", None)
            peft_config = peft.LoraConfig(
                task_type=TaskType.CAUSAL_LM if role != "critic" else TaskType.TOKEN_CLS,
                r=_value(lora, "rank"),
                lora_alpha=_value(lora, "alpha", _value(lora, "rank")),
                lora_dropout=_value(lora, "dropout", 0.0),
                target_modules=targets,
                bias=_value(lora, "bias", "none"),
            )
            model = get_peft_model(model, peft_config)

        if role == "ref":
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        if _value(fsdp_config, "enable_rank0_init", False):
            init_fn = get_init_fn(model, device="cuda")
            fsdp_kwargs = self._fsdp_kwargs(fsdp_config)
            fsdp_kwargs["param_init_fn"] = init_fn
        else:
            fsdp_kwargs = self._fsdp_kwargs(fsdp_config)

        model = FSDP(model, **fsdp_kwargs)
        model.train(role != "ref")

        if role == "actor":
            self.actor_module = model
            pair = self._create_optimizer(model, optim_config)
            self.actor_optimizer, self.actor_lr_scheduler = pair if pair else (None, None)
            self.actor_flops_counter = FlopsCounter(self.model_config)
        elif role == "critic":
            self.critic_module = model
            pair = self._create_optimizer(model, optim_config)
            self.critic_optimizer, self.critic_lr_scheduler = pair if pair else (None, None)
            self.critic_flops_counter = FlopsCounter(self.model_config)
        else:
            self.ref_module = model

        print_model_size(model, role)
        print_gpu_memory_usage(f"after building {role}")

    @_dispatch(Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_actor:
            self._build_model_optimizer(
                self.config.actor.model,
                self.config.actor.fsdp,
                self.config.actor.optim,
                _value(self.config.actor, "padding_free", False),
                "actor",
            )

        if self._has_critic:
            self._build_model_optimizer(
                self.config.critic.model,
                self.config.critic.fsdp,
                self.config.critic.optim,
                _value(self.config.critic, "padding_free", False),
                "critic",
            )

        if self._has_ref:
            ref_config = self.config.ref
            self._build_model_optimizer(
                ref_config.model,
                ref_config.fsdp,
                None,
                _value(ref_config, "padding_free", _value(self.config.actor, "padding_free", False)),
                "ref",
            )

        if self._has_rollout:
            actor = getattr(self, "actor_module", None)
            self.rollout = vLLMRollout(
                config=self.config.rollout,
                model_config=self.config.actor.model,
                tokenizer=self.tokenizer,
                processor=self.processor,
                model=actor,
            )
            self.sharding_manager = FSDPVLLMShardingManager(
                module=actor,
                inference_engine=self.rollout,
                device_mesh=self.device_mesh,
                offload_param=self._use_param_offload,
            )

        if self._use_param_offload and hasattr(self, "actor_module"):
            offload_fsdp_model(self.actor_module)
        if self._use_param_offload and hasattr(self, "critic_module"):
            offload_fsdp_model(self.critic_module)
        if self._use_optimizer_offload and hasattr(self, "actor_optimizer"):
            offload_fsdp_optimizer(self.actor_optimizer)
        if self._use_optimizer_offload and hasattr(self, "critic_optimizer"):
            offload_fsdp_optimizer(self.critic_optimizer)
        if self._use_ref_param_offload and hasattr(self, "ref_module"):
            offload_fsdp_model(self.ref_module)

    def _batch(self, data):
        return getattr(data, "batch", data)

    def _meta(self, data):
        return getattr(data, "meta_info", {})

    def _device_batch(self, data):
        batch = self._batch(data)
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _log_probs(self, model, batch, require_grad=False):
        context = nullcontext() if require_grad else torch.no_grad()
        with context:
            kwargs = {
                key: value
                for key, value in batch.items()
                if key in (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                )
            }
            output = model(**kwargs)
            logits = output.logits
            labels = batch.get("responses", batch.get("input_ids"))
            if labels.shape[-1] == logits.shape[-2]:
                labels = labels[..., 1:]
                logits = logits[..., :-1, :]
            else:
                logits = logits[..., -labels.shape[-1] :, :]
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(-1)
        return log_probs, entropy

    @_dispatch(Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.actor_module, device_id=torch.cuda.current_device())
        self.actor_module.eval()
        batch = self._device_batch(data)
        log_probs, _ = self._log_probs(self.actor_module, batch)
        self.actor_module.train()
        if self._use_param_offload:
            offload_fsdp_model(self.actor_module)
        return _make_proto({"old_log_probs": log_probs}, self._meta(data))

    @_dispatch(Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        if self._use_ref_param_offload:
            load_fsdp_model(self.ref_module, device_id=torch.cuda.current_device())
        self.ref_module.eval()
        batch = self._device_batch(data)
        log_probs, _ = self._log_probs(self.ref_module, batch)
        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_module)
        return _make_proto({"ref_log_probs": log_probs}, self._meta(data))

    @_dispatch(Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.critic_module, device_id=torch.cuda.current_device())
        self.critic_module.eval()
        batch = self._device_batch(data)
        with torch.no_grad():
            output = self.critic_module(
                **{
                    key: value
                    for key, value in batch.items()
                    if key in ("input_ids", "attention_mask", "position_ids", "pixel_values", "pixel_values_videos")
                }
            )
            values = output.logits.squeeze(-1)
            responses = batch.get("responses")
            if responses is not None:
                values = values[..., -responses.shape[-1] :]
        self.critic_module.train()
        if self._use_param_offload:
            offload_fsdp_model(self.critic_module)
        return _make_proto({"values": values}, self._meta(data))

    @_dispatch(Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.actor_module, device_id=torch.cuda.current_device())
        if self._use_optimizer_offload:
            load_fsdp_optimizer(self.actor_optimizer, device_id=torch.cuda.current_device())

        batch = self._device_batch(data)
        self.actor_module.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        log_probs, entropy = self._log_probs(self.actor_module, batch, require_grad=True)
        old_log_probs = batch.get("old_log_probs", log_probs.detach())
        advantages = batch.get("advantages", torch.ones_like(log_probs))
        ratio = torch.exp(log_probs - old_log_probs)
        clip_ratio = _value(self.config.actor, "clip_ratio", 0.2)
        objective = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages)
        loss = -objective.mean()

        entropy_coeff = _value(self.config.actor, "entropy_coeff", 0.0)
        if entropy_coeff:
            loss = loss - entropy_coeff * entropy.mean()

        loss.backward()
        max_norm = _value(self.config.actor, "max_grad_norm", 0.0)
        grad_norm = (
            self.actor_module.clip_grad_norm_(max_norm)
            if max_norm and max_norm > 0
            else torch.tensor(0.0, device=loss.device)
        )
        self.actor_optimizer.step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(self.actor_optimizer)
        if self._use_param_offload:
            offload_fsdp_model(self.actor_module)

        return _make_proto(
            {
                "actor_loss": loss.detach().reshape(1),
                "actor_entropy": entropy.detach().mean().reshape(1),
                "actor_grad_norm": torch.as_tensor(grad_norm, device=loss.device).reshape(1),
            }
        )

    @_dispatch(Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.critic_module, device_id=torch.cuda.current_device())
        if self._use_optimizer_offload:
            load_fsdp_optimizer(self.critic_optimizer, device_id=torch.cuda.current_device())

        batch = self._device_batch(data)
        self.critic_module.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        output = self.critic_module(
            **{
                key: value
                for key, value in batch.items()
                if key in ("input_ids", "attention_mask", "position_ids", "pixel_values", "pixel_values_videos")
            }
        )
        values = output.logits.squeeze(-1)
        targets = batch.get("returns", batch.get("advantages", values.detach()))
        values = values[..., -targets.shape[-1] :]
        loss = torch.nn.functional.mse_loss(values, targets)
        loss.backward()

        max_norm = _value(self.config.critic, "max_grad_norm", 0.0)
        grad_norm = (
            self.critic_module.clip_grad_norm_(max_norm)
            if max_norm and max_norm > 0
            else torch.tensor(0.0, device=loss.device)
        )
        self.critic_optimizer.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(self.critic_optimizer)
        if self._use_param_offload:
            offload_fsdp_model(self.critic_module)

        return _make_proto(
            {
                "critic_loss": loss.detach().reshape(1),
                "critic_grad_norm": torch.as_tensor(grad_norm, device=loss.device).reshape(1),
            }
        )

    @_dispatch(Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        if not self._has_rollout:
            raise RuntimeError("this worker has no rollout engine")
        if self._use_param_offload:
            load_fsdp_model(self.actor_module, device_id=torch.cuda.current_device())

        with self.sharding_manager:
            result = self.rollout.generate_sequences(prompts)

        if self._use_param_offload:
            offload_fsdp_model(self.actor_module)
        return result

    @_dispatch(Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path: str, global_step: int = 0, max_num_ckpt_to_keep: Optional[int] = None):
        module = getattr(self, "actor_module", getattr(self, "critic_module", None))
        optimizer = getattr(self, "actor_optimizer", getattr(self, "critic_optimizer", None))
        scheduler = getattr(self, "actor_lr_scheduler", getattr(self, "critic_lr_scheduler", None))
        if module is None:
            return
        manager = FSDPCheckpointManager(
            model=module,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            processing_class=getattr(self, "processor", getattr(self, "tokenizer", None)),
        )
        return manager.save_checkpoint(local_path, global_step, max_num_ckpt_to_keep)

    @_dispatch(Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path: str, del_local_after_load: bool = False):
        module = getattr(self, "actor_module", getattr(self, "critic_module", None))
        optimizer = getattr(self, "actor_optimizer", getattr(self, "critic_optimizer", None))
        scheduler = getattr(self, "actor_lr_scheduler", getattr(self, "critic_lr_scheduler", None))
        if module is None:
            return
        manager = FSDPCheckpointManager(
            model=module,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            processing_class=getattr(self, "processor", getattr(self, "tokenizer", None)),
        )
        return manager.load_checkpoint(local_path, del_local_after_load=del_local_after_load)