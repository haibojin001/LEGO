from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from transformers.models.llama.configuration_llama import LlamaConfig


def get_device_flops(unit: str = "T") -> float:
    def scale_flops(value: float, requested_unit: str) -> float:
        units = ["B", "K", "M", "G", "T", "P"]
        if value <= 0:
            return value

        position = 0
        while position < len(units) and units[position] != requested_unit:
            value /= 1000
            position += 1
        return value

    device = torch.cuda.get_device_name()
    peak_flops = float("inf")

    if "H100" in device or "H800" in device:
        peak_flops = 989e12
    elif "A100" in device or "A800" in device:
        peak_flops = 312e12
    elif "L40" in device:
        peak_flops = 181.05e12
    elif "L20" in device:
        peak_flops = 119.5e12
    elif "H20" in device:
        peak_flops = 148e12
    elif "910B" in device:
        peak_flops = 354e12

    return scale_flops(peak_flops, unit)


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)
    """

    def __init__(self, config: "LlamaConfig"):
        supported_estimators = {
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_llama_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_llama_flops,
            "qwen2_5_vl": self._estimate_llama_flops,
            "qwen3": self._estimate_llama_flops,
            "qwen3_vl": self._estimate_llama_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl_moe": self._estimate_qwen2_moe_flops,
        }

        if config.model_type not in supported_estimators:
            print(f"Only support {supported_estimators.keys()}, but got {config.model_type}. MFU will always be zero.")

        self.config = getattr(config, "text_config", config)
        self._estimate_flops = supported_estimators.get(config.model_type, self._estimate_unknown_flops)

    def _estimate_unknown_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        return 0

    def _estimate_llama_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        mlp_parameters = hidden_size * intermediate_size * 3
        attention_parameters = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        embedding_output_parameters = vocab_size * hidden_size * 2

        dense_parameters = (
            (mlp_parameters + attention_parameters) * num_hidden_layers
            + embedding_output_parameters
        )
        dense_flops = 6 * dense_parameters * tokens_sum

        sequence_square_sum = 0
        for sequence_length in batch_seqlens:
            sequence_square_sum += sequence_length * sequence_length

        attention_flops = (
            12
            * sequence_square_sum
            * head_dim
            * num_attention_heads
            * num_hidden_layers
        )

        total_flops = dense_flops + attention_flops
        return total_flops * (1.0 / delta_time) / 1e12

    def _estimate_qwen2_moe_flops(
        self, tokens_sum: int, batch_seqlens: List[int], delta_time: float
    ) -> float:
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        moe_intermediate_size = config.moe_intermediate_size
        moe_topk = config.num_experts_per_tok
        num_experts = config.num_experts

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        moe_mlp_parameters = (
            hidden_size * moe_topk * moe_intermediate_size * 3
            + hidden_size * num_experts
        )
        attention_parameters = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        embedding_output_parameters = vocab_size * hidden_size * 2

        dense_parameters = (
            (moe_mlp_parameters + attention_parameters) * num_hidden_layers
            + embedding_output_parameters
        )
        dense_flops = 6 * dense_parameters * tokens_sum

        sequence_square_sum = 0
        for sequence_length in batch_seqlens:
            sequence_square_sum += sequence_length * sequence_length

        attention_flops = (
            12
            * sequence_square_sum
            * head_dim
            * num_attention_heads
            * num_hidden_layers
        )

        total_flops = dense_flops + attention_flops
        return total_flops * (1.0 / delta_time) / 1e12

    def estimate_flops(self, batch_seqlens: List[int], delta_time: float) -> Tuple[float, float]:
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        estimated_flops = self._estimate_flops(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops