import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    if processor is None or not hasattr(processor, "image_token"):
        return None

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    return {image_token_id: -100}


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    images = []
    videos = []

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(
                process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
            )

    if images:
        return {"image": images}
    if videos:
        return {"video": videos}
    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        super().__init__()

        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.return_video_metadata = (
            processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        )
        self.use_tqdm = self.rank == 0 and not config.disable_tqdm

        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        self.lora_kwargs = kwargs.pop("lora_kwargs", {})

        engine_kwargs = {}
        if processor is not None:
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        previous_values = {}
        for key, value in kwargs.items():
            if hasattr(self.sampling_params, key):
                previous_values[key] = getattr(self.sampling_params, key)
                setattr(self.sampling_params, key, value)

        try:
            yield
        finally:
            for key, value in previous_values.items():
                setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is None:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)}
                for raw_prompt_ids in batch_raw_prompt_ids
            ]
        else:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )

        lora_requests = None
        if self.lora_kwargs:
            lora_ids = list(self.inference_engine.llm_engine.list_loras())
            if lora_ids:
                lora_id = lora_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/simon-stub-path",
                    )
                ] * batch_size

        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=self.use_tqdm,
                lora_request=lora_requests,
            )

        response_token_ids = [
            output.token_ids
            for completion in completions
            for output in completion.outputs
        ]

        response_tensors = [
            torch.tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
            for token_ids in response_token_ids
        ]
        if response_tensors:
            response = torch.nn.utils.rnn.pad_sequence(
                response_tensors,
                batch_first=True,
                padding_value=self.pad_token_id,
            )
        else:
            response = torch.empty(
                (0, 0),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        response = VF.pad_sequence_to_length(
            response,
            self.config.response_length,
            self.pad_token_id,
        )

        num_samples = self.config.n
        if num_samples > 1:
            batch_size *= num_samples
            input_ids = _repeat_interleave(input_ids, num_samples)
            attention_mask = _repeat_interleave(attention_mask, num_samples)
            position_ids = _repeat_interleave(position_ids, num_samples)
            non_tensor_batch = {
                key: _repeat_interleave(value, num_samples)
                for key, value in non_tensor_batch.items()
            }

        sequence = torch.cat([input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(
            1,
            response_length + 1,
            device=position_ids.device,
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size,
                position_ids.size(1),
                -1,
            )

        position_ids = torch.cat(
            [position_ids, position_ids[..., -1:] + delta_position_id],
            dim=-1,
        )

        response_attention_mask = VF.get_eos_mask(
            response_id=response,
            eos_token_id=eos_token_id,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response,
                "input_ids": sequence,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=prompts.meta_info,
        )