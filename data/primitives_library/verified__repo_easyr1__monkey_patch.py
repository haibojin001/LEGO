from importlib import import_module as _load_module

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.py_functional import is_transformers_version_greater_than
from .transformers.flash_attention_utils import flash_attention_forward


SUPPORTED_MODEL_TYPE = (
    "llama",
    "gemma",
    "gemma2",
    "mistral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
)

QWEN2_VL_MODELS = ("qwen2_vl", "qwen2_5_vl")
QWEN3_VL_MODELS = ("qwen3_vl", "qwen3_vl_moe")


def apply_ulysses_patch(model_type: str) -> None:
    if not is_transformers_version_greater_than("4.54.0"):
        raise RuntimeError("Only support transformers >= 4.54.0.")

    if model_type not in SUPPORTED_MODEL_TYPE:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

    if model_type in QWEN2_VL_MODELS:
        qwen2_vl = _load_module("transformers.models.qwen2_vl.modeling_qwen2_vl")
        qwen2_5_vl = _load_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
        patched_forwards = _load_module(".transformers.qwen2_vl", package=__package__)

        qwen2_vl.Qwen2VLModel.forward = patched_forwards.qwen2_vl_base_forward
        qwen2_5_vl.Qwen2_5_VLModel.forward = patched_forwards.qwen2_vl_base_forward
        qwen2_vl.Qwen2VLForConditionalGeneration.forward = patched_forwards.qwen2_vl_model_forward
        qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = patched_forwards.qwen2_vl_model_forward
    elif model_type in QWEN3_VL_MODELS:
        qwen3_vl = _load_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
        qwen3_vl_moe = _load_module("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
        patched_forwards = _load_module(".transformers.qwen3_vl", package=__package__)

        qwen3_vl.Qwen3VLModel.forward = patched_forwards.qwen3_vl_base_forward
        qwen3_vl_moe.Qwen3VLMoeModel.forward = patched_forwards.qwen3_vl_base_forward
        qwen3_vl.Qwen3VLForConditionalGeneration.forward = patched_forwards.qwen3_vl_model_forward
        qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration.forward = patched_forwards.qwen3_vl_model_forward