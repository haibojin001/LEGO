import os
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import compute_value_loss
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOCritic
from .config import CriticConfig

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOCritic"]


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(
        self,
        config: CriticConfig,
        critic_module: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
    ):
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        response_length = micro_batch["responses"].size(-1)

        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            collated_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {
                key: torch.cat(chunks, dim=0) for key, chunks in collated_inputs.items()
            }
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            flat_input_ids, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1),
                attention_mask,
            )
            flat_input_ids = flat_input_ids.transpose(0, 1)

            if position_ids.dim() == 3:
                flat_position_ids = index_first_axis(
                    rearrange(position_ids, "c b s ... -> (b s) c ..."),
                    indices,
                )
                flat_position_ids = flat_position_ids.transpose(0, 1).unsqueeze(1)
            else:
                flat_position_ids = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

            if self.config.ulysses_size > 1:
                flat_input_ids, flat_position_ids, padding_size = ulysses_pad_and_slice_inputs(
                    flat_input_ids,
                    flat_position_ids,
                    sp_size=self.config.ulysses_size,
                )

            outputs = self.critic_module(
                input_ids=flat_input_ids,
                attention_mask=None,
                position_ids=flat_position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            flat_values = outputs.logits.squeeze(0)

            if self.config.ulysses_size > 1:
                flat_values = gather_outputs_and_unpad(
                    flat_values,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=padding_size,
                )

            values = pad_input(
                flat_values,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            ).squeeze(-1)
            values = values[:, -response_length - 1 : -1]
        else:
            outputs = self.critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            values = outputs.logits[:, -response_length - 1 : -1].squeeze(-1)

        return values

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_module.parameters(),
                max_norm=self.config.max_grad_norm,
            )

        if torch.isfinite(grad_norm):
            self.critic_optimizer.step()
        else:
            print("Gradient norm is not finite. Skip update.")

        self.critic_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()

        tensor_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
        ]
        non_tensor_keys = ["multi_modal_inputs"]
        data = data.select(tensor_keys, non_tensor_keys)

        if self.config.dynamic_batching:
            max_token_len = (
                self.config.micro_batch_size_per_device_for_experience
                * data.batch["input_ids"].size(-1)
            )
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
            )
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        values_list = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute values", position=1)

        for micro_batch in micro_batches:
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            values_list.append(self._forward_micro_batch(inputs))

        values = torch.concat(values_list, dim=0)

        if self.config.dynamic_batching:
            values = restore_dynamic_batch(values, batch_idx_list)

        return values * data.batch["response_mask"]

    def update_critic(self, data: DataProto) -> dict[str, Any]:
        self.critic_module.train()

        tensor_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "values",
            "returns",
        ]
        non_tensor_keys = ["multi_modal_inputs"]
        mini_batches = data.select(tensor_keys, non_tensor_keys).split(
            self.config.global_batch_size_per_device
        )

        metrics = defaultdict(list)

        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = (
                        self.config.micro_batch_size_per_device_for_update * max_input_len
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                    )
                else:
                    micro_batches = mini_batch.split(
                        self.config.micro_batch_size_per_device_for_update
                    )

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update critic", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                    }
                    response_mask = model_inputs["response_mask"]
                    old_values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    value_predictions = self._forward_micro_batch(model_inputs)
                    value_loss, value_metrics = compute_value_loss(
                        value_predictions,
                        old_values,
                        returns,
                        response_mask,
                        cliprange_value=self.config.cliprange_value,
                    )

                    scaled_loss = (
                        value_loss
                        * torch.sum(response_mask)
                        * self.world_size
                        / total_response_tokens
                    )
                    scaled_loss.backward()
                    append_to_dict(metrics, value_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {"critic/grad_norm": grad_norm.detach().item()},
                )

        return {
            name: sum(values) / len(values)
            for name, values in metrics.items()
        }