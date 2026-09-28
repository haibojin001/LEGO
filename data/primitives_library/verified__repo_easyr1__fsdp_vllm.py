import inspect
import re
import time
from dataclasses import asdict
from typing import Iterable, Union

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import (
    load_fsdp_model,
    load_fsdp_submodule,
    offload_fsdp_model,
    offload_fsdp_submodule,
)
from ...utils.model_utils import print_gpu_memory_usage
from ...utils.vllm_utils import TensorLoRARequest
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False
        self.is_lora = isinstance(module._fsdp_wrapped_module, PeftModel)

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        self.freed_bytes = 0

        self.torch_random_states = torch.cuda.get_rng_state()
        dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(dp_rank + 1000)
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def _rename_weight_keys(
        self,
        actor_weights: dict[str, Union[torch.Tensor, DTensor]],
        model: PreTrainedModel,
    ):
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_mapping = {
            mapped_name: original_name
            for original_name, mapped_name in model._checkpoint_conversion_mapping.items()
        }
        result = {}

        for name, weight in actor_weights.items():
            converted_name = name
            for pattern, replacement in reverse_mapping.items():
                replacement = replacement.lstrip("^")
                replacement = re.sub(r"\(.*\)", "", replacement)
                converted_name, replacements = re.subn(
                    pattern, replacement, converted_name
                )
                if replacements > 0:
                    break
            result[converted_name] = weight

        return result

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            yield name, tensor

    def _collect_lora_weights(self) -> dict:
        lora_weights = {}
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)

        for name, submodule in self.module.named_modules():
            if not name.rsplit("layers.", 1)[-1].isdigit():
                continue

            if self.use_param_offload:
                load_fsdp_submodule(submodule)

            layer_prefix = name.replace(
                "_fsdp_wrapped_module.base_model.model.", "base_model.model."
            )
            layer_state_dict = get_model_state_dict(submodule)
            adapter_state_dict = get_peft_model_state_dict(
                peft_model, state_dict=layer_state_dict
            )

            for adapter_name, adapter_weight in adapter_state_dict.items():
                full_name = f"{layer_prefix}.{adapter_name}"
                if isinstance(adapter_weight, DTensor):
                    lora_weights[full_name] = (
                        adapter_weight.full_tensor().detach().cpu()
                    )
                else:
                    lora_weights[full_name] = adapter_weight.detach().cpu()

            submodule._is_root = False

            if self.use_param_offload:
                offload_fsdp_submodule(submodule)

            torch.cuda.empty_cache()

        return lora_weights

    def _sync_weight_to_vllm(self):
        if self.use_param_offload and not self.is_lora:
            load_fsdp_model(self.module)

        if self.is_lora:
            peft_config = self.module._fsdp_wrapped_module.peft_config.get(
                "default", None
            )
            actor_weights = self._collect_lora_weights()
        else:
            actor_weights = get_model_state_dict(self.module)
            actor_weights = self._rename_weight_keys(
                actor_weights, self.module._fsdp_wrapped_module
            )

        print_gpu_memory_usage("After gather model weights in sharding manager")

        vllm_model = (
            self.inference_engine.llm_engine.model_executor.driver_worker.worker
            .model_runner.model
        )

        if self.is_lora:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=str(lora_int_id),
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=actor_weights,
            )
            self.inference_engine.llm_engine.add_lora(lora_request)
            print_gpu_memory_usage("After load LoRA weights in sharding manager")
        else:
            vllm_model.load_weights(self._make_weight_iterator(actor_weights))

        del actor_weights

        if self.use_param_offload and not self.is_lora:
            offload_fsdp_model(self.module)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in sharding manager")

    def load_vllm_and_sync_weights(self):
        torch.cuda.empty_cache()
        assert not self.loaded, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in sharding manager")

        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        assert self.loaded, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After vllm offload in sharding manager")

        if self.device_mesh is not None:
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        return all_gather_data_proto(data, process_group=self.tp_group)

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def __enter__(self):
        self.load_vllm_and_sync_weights()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.offload_vllm()