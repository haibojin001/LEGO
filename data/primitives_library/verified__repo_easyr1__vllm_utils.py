from importlib.metadata import version
from typing import List

from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def load_tensor_aware_adapter(
            self, lora_request: TensorLoRARequest
        ) -> LoRAModel:
            available_modules = self._adapter_manager.supported_lora_modules
            packed_modules = self._adapter_manager.packed_modules_mapping

            required_modules: List[str] = []
            for module_name in available_modules:
                if module_name in packed_modules:
                    required_modules.extend(packed_modules[module_name])
                else:
                    required_modules.append(module_name)
            required_modules = list(set(required_modules))

            from vllm.lora.peft_helper import PEFTHelper

            tensor_weights = None
            if isinstance(lora_request, TensorLoRARequest):
                tensor_weights = lora_request.lora_tensors
                peft = PEFTHelper.from_dict(lora_request.peft_config)
            else:
                adapter_directory = get_adapter_absolute_path(
                    lora_request.lora_path
                )
                peft = PEFTHelper.from_local_dir(
                    adapter_directory,
                    self.max_position_embeddings,
                )

            peft.validate_legal(self.lora_config)

            translation = None
            base_model = self._adapter_manager.model
            if (
                hasattr(base_model, "hf_to_vllm_mapper")
                and base_model.hf_to_vllm_mapper is not None
            ):
                translation = base_model.hf_to_vllm_mapper

            embedding_padding = (
                self.vocab_size + self.lora_config.lora_extra_vocab_size
            )

            if isinstance(lora_request, TensorLoRARequest):
                loaded_lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=tensor_weights,
                    peft_helper=peft,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=embedding_padding,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=translation,
                )
            else:
                loaded_lora = self._lora_model_cls.from_local_checkpoint(
                    adapter_directory,
                    required_modules,
                    peft_helper=peft,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    target_embedding_padding=embedding_padding,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=translation,
                )

            if (
                loaded_lora.extra_vocab_size
                > self.lora_config.lora_extra_vocab_size
            ):
                raise ValueError(
                    f"LoRA added vocab size {loaded_lora.extra_vocab_size} "
                    f"is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )

            return loaded_lora

        setattr(
            LRUCacheWorkerLoRAManager,
            "_load_adapter",
            load_tensor_aware_adapter,
        )

        if vs.parse(version("vllm")).base_version == "0.11.0":
            from vllm.model_executor.models.module_mapping import MultiModelKeys
            from vllm.model_executor.models.qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )

            def qwen3_vl_mm_mapping(self) -> MultiModelKeys:
                return MultiModelKeys.from_string_field(
                    language_model="language_model",
                    connector="visual.merger.",
                    tower_model="visual.",
                )

            setattr(
                Qwen3VLForConditionalGeneration,
                "get_mm_mapping",
                qwen3_vl_mm_mapping,
            )