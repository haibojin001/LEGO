from __future__ import annotations

import base64
import io
import json
import zipfile
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from agentlightning.verl.agl_rollout_manager import CompletedRollout

_TRACE_MERGE_MISMATCH_WANDB_LIMIT = 100
_TRACE_MERGE_MISMATCH_TEXT_LIMIT = 4000
_ROLLOUT_TRAJECTORY_WANDB_LIMIT = 24

_TRACE_MERGE_MISMATCH_COLUMNS = [
    "global_steps",
    "rollout_id",
    "data_id",
    "turn_index",
    "template_mismatch",
    "retoken_mismatch",
    "others_mismatch",
    "prompt_length",
    "response_length",
    "previous_trace_length",
    "current_trace_length",
    "previous_trace",
    "current_trace",
]

_ROLLOUT_TRAJECTORY_COLUMNS = [
    "global_steps",
    "trajectory_artifact",
    "trajectory_artifact_path",
    "row_count",
]

_MROPE_PROCESSOR_TAGS = ("Qwen2VL", "Qwen2_5_VL", "Qwen3VL", "Qwen3_5")


def ids_startswith(full_ids: list[int], prefix_ids: list[int]) -> bool:
    return full_ids[: len(prefix_ids)] == prefix_ids


def _decode_token_ids(tokenizer: Any | None, ids: list[int]) -> str:
    if tokenizer is not None:
        try:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except TypeError:
            text = tokenizer.decode(ids)
        except Exception:
            text = " ".join(str(i) for i in ids)
    else:
        text = " ".join(str(i) for i in ids)
    return text


def _decode_trace_text(tokenizer: Any | None, ids: list[int]) -> str:
    text = _decode_token_ids(tokenizer, ids)
    if len(text) > _TRACE_MERGE_MISMATCH_TEXT_LIMIT:
        truncated = len(text) - _TRACE_MERGE_MISMATCH_TEXT_LIMIT
        return text[:_TRACE_MERGE_MISMATCH_TEXT_LIMIT] + f"\n...[truncated {truncated} chars]"
    return text


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict) and isinstance(value.get("token_ids"), list):
        return value["token_ids"]
    return []


def _artifact_safe_name(value: Any) -> str:
    text = str(value)
    safe = "".join(
        char if char.isascii() and (char.isalnum() or char in {"-", "_", "."}) else "_"
        for char in text
    )
    return safe or "unknown"


def _build_compact_rollout_trajectory_records(
    rollouts: list[CompletedRollout],
    *,
    tokenizer: Any | None,
    reward_fillna_value: float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sorted_rollouts = sorted(rollouts, key=lambda rollout: (rollout.step, rollout.sample_idx_in_step))
    for rollout in sorted_rollouts:
        if limit is not None and len(records) >= limit:
            break
        if not rollout.triplets:
            continue
        last_triplet = rollout.triplets[-1]
        records.append(
            {
                "rollout_id": rollout.rollout_id,
                "reward": rollout.final_reward if rollout.final_reward is not None else reward_fillna_value,
                "prompt": _decode_token_ids(tokenizer, _token_ids(last_triplet.prompt)),
                "response": _decode_token_ids(tokenizer, _token_ids(last_triplet.response)),
            }
        )
    return records


def _build_zipped_jsonl(records: list[dict[str, Any]], jsonl_name: str) -> bytes:
    jsonl_text = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(jsonl_name, jsonl_text.encode("utf-8"))
    return buffer.getvalue()


def _upload_trace_merge_mismatches_to_wandb(rows: list[dict[str, Any]], global_steps: int) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        table = wandb.Table(columns=cast(list[str | int], _TRACE_MERGE_MISMATCH_COLUMNS))
        for row in rows:
            table.add_data(*(row.get(column) for column in _TRACE_MERGE_MISMATCH_COLUMNS))
        wandb.log({"training/trace_merge_mismatches": table}, step=global_steps)
    except Exception as exc:
        print(f"Warning: failed to upload trace merge mismatches to wandb: {exc}")


def _upload_compact_rollout_trajectories_to_wandb(
    records: list[dict[str, Any]],
    global_steps: int,
    *,
    is_validation: bool = False,
) -> None:
    split = "validation" if is_validation else "train"
    try:
        import wandb

        if wandb.run is None:
            return
        run = wandb.run
        artifact_type = f"{split}_trajectories"
        artifact_path = f"step_{global_steps}/{artifact_type}.jsonl.zip"
        artifact_name = (
            f"{split}-trajectories-{_artifact_safe_name(getattr(run, 'id', None) or 'run')}-step-{global_steps}"
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata={"global_steps": global_steps, "row_count": len(records), "format": "jsonl.zip"},
        )
        with artifact.new_file(artifact_path, mode="wb") as trajectory_file:
            trajectory_file.write(_build_zipped_jsonl(records, f"{artifact_type}.jsonl"))
        run.log_artifact(artifact)

        table = wandb.Table(columns=cast(list[str | int], _ROLLOUT_TRAJECTORY_COLUMNS))
        table.add_data(global_steps, artifact_name, artifact_path, len(records))
        table_key = "val/rollout_trajectories" if is_validation else "training/rollout_trajectories"
        wandb.log({table_key: table}, step=global_steps)
    except Exception as exc:
        print(f"Warning: failed to upload {split} trajectories to wandb: {exc}")


def get_left_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    seq_len = len(ids)
    if seq_len >= max_length:
        return ids[-max_length:], [1] * max_length
    pad_len = max_length - seq_len
    return [pad_token_id] * pad_len + ids, [0] * pad_len + [1] * seq_len


def get_right_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    seq_len = len(ids)
    if seq_len >= max_length:
        return ids[:max_length], [1] * max_length
    pad_len = max_length - seq_len
    return ids + [pad_token_id] * pad_len, [1] * seq_len + [0] * pad_len


def _is_mrope_processor(processor: Any) -> bool:
    if processor is None:
        return False
    if getattr(processor, "get_rope_index", None) is not None:
        return True
    names = [processor.__class__.__name__]
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        names.append(image_processor.__class__.__name__)
    return any(tag in name for name in names for tag in _MROPE_PROCESSOR_TAGS)


def _load_pil_image(url: str) -> Any:
    from PIL import Image

    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[len("file://") :]).convert("RGB")
    if url.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"[multimodal-patch] remote URL is not an image (content-type={content_type!r})")
        max_bytes = 50 * 1024 * 1024
        if len(response.content) > max_bytes:
            raise ValueError(f"[multimodal-patch] remote image exceeds {max_bytes} bytes")
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    raise ValueError(f"[multimodal-patch] unsupported image url scheme: {url[:64]}")


def _build_multi_modal_inputs(processor: Any, image_urls: list[str]) -> dict[str, Any]:
    if not image_urls:
        return {}
    images = [_load_pil_image(url) for url in image_urls]
    processed = processor(images=images, return_tensors="pt")
    if hasattr(processed, "data"):
        processed = processed.data
    if not isinstance(processed, dict):
        processed = dict(processed)
    return {
        key: value
        for key, value in processed.items()
        if key not in {"input_ids", "attention_mask", "token_type_ids"}
    }


def _compute_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, Any] | None = None,
) -> torch.Tensor:
    if not multi_modal_inputs or not _is_mrope_processor(processor):
        return attention_mask.cumsum(dim=-1).clamp_min(1).sub(1).long()

    get_rope_index = getattr(processor, "get_rope_index", None)
    if get_rope_index is None:
        return attention_mask.cumsum(dim=-1).clamp_min(1).sub(1).long()

    kwargs = dict(multi_modal_inputs)
    try:
        result = get_rope_index(input_ids, attention_mask=attention_mask, **kwargs)
    except TypeError:
        try:
            result = get_rope_index(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        except Exception:
            result = None
    except Exception:
        result = None

    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, torch.Tensor):
        if result.ndim == 2:
            result = result.unsqueeze(0).expand(3, -1, -1)
        if result.ndim == 3 and result.shape[0] == 3:
            result = result.unsqueeze(0)
        if result.ndim == 4:
            return result.squeeze(0).long()
    plain = attention_mask.cumsum(dim=-1).clamp_min(1).sub(1).long()
    return plain.unsqueeze(0).expand(3, -1, -1)


def _value_images(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    for key in ("image_urls", "images", "image_url"):
        images = value.get(key)
        if isinstance(images, str):
            return [images]
        if isinstance(images, list):
            return [image for image in images if isinstance(image, str)]
    return []


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _triplet_trace_ids(triplet: Any) -> tuple[list[int], list[int]]:
    prompt = _token_ids(_get_attr_or_key(triplet, "prompt", {}))
    response = _token_ids(_get_attr_or_key(triplet, "response", {}))
    return prompt, response


def _make_dataproto(
    batch: TensorDict,
    non_tensor_batch: dict[str, Any],
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    meta_info = meta_info or {}
    try:
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    except TypeError:
        try:
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        except TypeError:
            return DataProto(batch=batch)


def completed_rollouts_to_dataproto(
    rollouts: list[CompletedRollout],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int | None = None,
    reward_fillna_value: float = 0.0,
    processor: Any | None = None,
) -> DataProto:
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = int(pad_token_id)

    rows: list[dict[str, Any]] = []
    for rollout in sorted(rollouts, key=lambda item: (item.step, item.sample_idx_in_step)):
        triplets = _get_attr_or_key(rollout, "triplets", []) or []
        if not triplets:
            continue
        prompt_ids, response_ids = _triplet_trace_ids(triplets[-1])
        if not response_ids:
            continue
        images = _value_images(_get_attr_or_key(triplets[-1], "prompt", {}))
        reward = _get_attr_or_key(rollout, "final_reward")
        rows.append(
            {
                "prompt": prompt_ids,
                "response": response_ids,
                "reward": reward_fillna_value if reward is None else reward,
                "uid": str(_get_attr_or_key(rollout, "rollout_id", len(rows))),
                "images": images,
            }
        )

    if not rows:
        empty = TensorDict(
            {
                "prompts": torch.empty((0, max_prompt_length), dtype=torch.long),
                "responses": torch.empty((0, max_response_length), dtype=torch.long),
                "input_ids": torch.empty((0, max_prompt_length + max_response_length), dtype=torch.long),
                "attention_mask": torch.empty((0, max_prompt_length + max_response_length), dtype=torch.long),
                "position_ids": torch.empty((0, max_prompt_length + max_response_length), dtype=torch.long),
            },
            batch_size=[0],
        )
        return _make_dataproto(empty, {"reward": np.asarray([], dtype=np.float32), "uid": np.asarray([], dtype=object)})

    prompts: list[list[int]] = []
    prompt_masks: list[list[int]] = []
    responses: list[list[int]] = []
    response_masks: list[list[int]] = []

    for row in rows:
        prompt, prompt_mask = get_left_padded_ids_and_attention_mask(row["prompt"], max_prompt_length, pad_token_id)
        response, response_mask = get_right_padded_ids_and_attention_mask(
            row["response"], max_response_length, pad_token_id
        )
        prompts.append(prompt)
        prompt_masks.append(prompt_mask)
        responses.append(response)
        response_masks.append(response_mask)

    prompt_tensor = torch.tensor(prompts, dtype=torch.long)
    response_tensor = torch.tensor(responses, dtype=torch.long)
    attention_mask = torch.tensor([a + b for a, b in zip(prompt_masks, response_masks)], dtype=torch.long)
    input_ids = torch.cat((prompt_tensor, response_tensor), dim=-1)
    position_ids = attention_mask.cumsum(dim=-1).clamp_min(1).sub(1).long()

    batch = TensorDict(
        {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=[len(rows)],
    )
    non_tensor_batch: dict[str, Any] = {
        "reward": np.asarray([row["reward"] for row in rows], dtype=np.float32),
        "uid": np.asarray([row["uid"] for row in rows], dtype=object),
    }
    return _make_dataproto(batch, non_tensor_batch)


def rollouts_to_dataproto(
    rollouts: list[CompletedRollout],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int | None = None,
    reward_fillna_value: float = 0.0,
    processor: Any | None = None,
) -> DataProto:
    return completed_rollouts_to_dataproto(
        rollouts,
        tokenizer,
        max_prompt_length,
        max_response_length,
        pad_token_id,
        reward_fillna_value,
        processor,
    )