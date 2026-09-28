import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from ray.experimental.tqdm_ray import tqdm
except Exception:
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(iterable, **kwargs):
            return iterable

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    index_first_axis = None
    pad_input = None
    unpad_input = None


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(
                VF.log_probs_from_logits,
                dynamic=True,
            )
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        input_ids = micro_batch["input_ids"]
        batch_size, sequence_length = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)

        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            collated = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {
                key: torch.cat(values, dim=0) for key, values in collated.items()
            }
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            if unpad_input is None or index_first_axis is None or pad_input is None:
                raise ImportError("flash_attn is required for padding_free execution")

            input_ids_unpadded, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1),
                attention_mask,
            )
            input_ids_unpadded = input_ids_unpadded.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_unpadded = index_first_axis(
                    rearrange(position_ids, "c b s ... -> (b s) c ..."),
                    indices,
                ).transpose(0, 1).unsqueeze(1)
            else:
                position_ids_unpadded = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

            shifted_input_ids = torch.roll(
                input_ids_unpadded,
                shifts=-1,
                dims=1,
            )

            padding_size = 0
            if self.config.ulysses_size > 1:
                input_ids_unpadded, position_ids_unpadded, padding_size = (
                    ulysses_pad_and_slice_inputs(
                        input_ids_unpadded,
                        position_ids_unpadded,
                        sp_size=self.config.ulysses_size,
                    )
                )
                shifted_input_ids, _, _ = ulysses_pad_and_slice_inputs(
                    shifted_input_ids,
                    None,
                    sp_size=self.config.ulysses_size,
                )

            shifted_input_ids = shifted_input_ids.squeeze(0)

            output = self.actor_module(
                input_ids=input_ids_unpadded,
                attention_mask=None,
                position_ids=position_ids_unpadded,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = output.logits.squeeze(0)
            logits.div_(temperature)

            log_probs = self.log_probs_from_logits(
                logits=logits,
                labels=shifted_input_ids,
            )

            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=padding_size,
                )

            log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1),
                indices=indices,
                batch=batch_size,
                seqlen=sequence_length,
            ).squeeze(-1)
            return log_probs[:, -response_length - 1 : -1]

        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        return self.log_probs_from_logits(logits, responses)

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_module.parameters(),
                max_norm=self.config.max_grad_norm,
            )

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        tensor_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
        ]
        data = data.select(tensor_keys, ["multi_modal_inputs"])

        batch_idx_list = None
        if self.config.dynamic_batching:
            maximum_tokens = (
                self.config.micro_batch_size_per_device_for_experience
                * data.batch["input_ids"].size(-1)
            )
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=maximum_tokens,
            )
        else:
            micro_batches = data.split(
                self.config.micro_batch_size_per_device_for_experience
            )

        if self.rank == 0:
            micro_batches = tqdm(
                micro_batches,
                desc="Compute log probs",
                position=1,
            )

        results = []
        for micro_batch in micro_batches:
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            results.append(
                self._forward_micro_batch(inputs, temperature=temperature)
            )

        log_probs = torch.cat(results, dim=0)
        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        tensor_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            tensor_keys.append("ref_log_probs")

        data = data.select(tensor_keys, ["multi_modal_inputs"])

        if self.config.dynamic_batching:
            maximum_tokens = (
                self.config.micro_batch_size_per_device_for_update
                * data.batch["input_ids"].size(-1)
            )
            micro_batches, _ = prepare_dynamic_batch(
                data,
                max_token_len=maximum_tokens,
            )
        else:
            micro_batches = data.split(
                self.config.micro_batch_size_per_device_for_update
            )

        if self.rank == 0:
            micro_batches = tqdm(
                micro_batches,
                desc="Update policy",
                position=1,
            )

        metrics: dict[str, Any] = defaultdict(list)
        self.actor_optimizer.zero_grad()

        for micro_batch in micro_batches:
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            old_log_probs = micro_batch.batch["old_log_probs"]
            advantages = micro_batch.batch["advantages"]
            response_length = micro_batch.batch["responses"].size(-1)
            response_mask = micro_batch.batch["attention_mask"][:, -response_length:]

            log_probs = self._forward_micro_batch(
                inputs,
                temperature=temperature,
            )

            pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(
                old_log_prob=old_log_probs,
                log_prob=log_probs,
                advantages=advantages,
                response_mask=response_mask,
                cliprange=self.config.clip_ratio,
            )

            loss = pg_loss
            metric_values = {
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                "actor/ppo_kl": ppo_kl.detach().item(),
            }

            if self.config.use_kl_loss:
                ref_log_probs = micro_batch.batch["ref_log_probs"]
                kl_values = compute_kl(
                    log_probs,
                    ref_log_probs,
                    kl_penalty=self.config.kl_loss_type,
                )
                kl_loss = average_loss(kl_values, response_mask)
                loss = loss + self.config.kl_loss_coef * kl_loss
                metric_values["actor/kl_loss"] = kl_loss.detach().item()

            (loss / self.config.gradient_accumulation).backward()
            append_to_dict(metrics, metric_values)

        grad_norm = self._optimizer_step()

        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_norm, op=dist.ReduceOp.AVG)

        append_to_dict(
            metrics,
            {"actor/grad_norm": grad_norm.detach().item()},
        )
        return metrics