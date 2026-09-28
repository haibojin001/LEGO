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


def _build_multi_modal_inputs(processor: Any, image_urls: list[str]) -> dict[str, Any] | None:
    if processor is None or not image_urls:
        return None
    images = [_load_pil_image(url) for url in image_urls]
    image_processor = getattr(processor, "image_processor", processor)
    try:
        processed = image_processor(images=images, return_tensors="pt")
    except TypeError:
        processed = processor(images=images, return_tensors="pt")

    if hasattr(processed, "data"):
        processed = processed.data
    if not isinstance(processed, dict):
        return None

    result: dict[str, Any] = {}
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in processed:
            result[key] = processed[key]
    return result or None


def _compute_mrope_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, Any] | None,
) -> torch.Tensor | None:
    if processor is None or not _is_mrope_processor(processor):
        return None
    get_rope_index = getattr(processor, "get_rope_index", None)
    if get_rope_index is None:
        return None

    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if multi_modal_inputs:
        kwargs.update(multi_modal_inputs)

    try:
        result = get_rope_index(**kwargs)
    except TypeError:
        try:
            result = get_rope_index(input_ids, attention_mask, **(multi_modal_inputs or {}))
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        return None
    if result.ndim == 2:
        result = result.unsqueeze(0)
    return result


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _get_image_urls(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    candidates = (
        value.get("image_urls"),
        value.get("images"),
        value.get("image_url"),
        value.get("image"),
    )
    for candidate in candidates:
        if isinstance(candidate, str):
            return [candidate]
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, str)]
    return []


def _make_dataproto(batch: dict[str, torch.Tensor], non_tensor_batch: dict[str, Any]) -> DataProto:
    tensordict = TensorDict(batch, batch_size=[next(iter(batch.values())).shape[0]])
    try:
        return DataProto(batch=tensordict, non_tensor_batch=non_tensor_batch)
    except TypeError:
        try:
            return DataProto.from_dict(tensors=batch, non_tensors=non_tensor_batch)
        except TypeError:
            return DataProto(tensordict, non_tensor_batch)


def convert_rollouts_to_data_proto(
    rollouts: list[CompletedRollout],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int | None = None,
    reward_fillna_value: float = 0.0,
    processor: Any | None = None,
    global_steps: int | None = None,
    is_validation: bool = False,
) -> DataProto:
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []

    for rollout in rollouts:
        triplets = _get_value(rollout, "triplets", []) or []
        reward = _get_value(rollout, "final_reward", None)
        if reward is None:
            reward = reward_fillna_value
        rollout_id = _get_value(rollout, "rollout_id", None)
        previous_trace: list[int] = []

        for turn_index, triplet in enumerate(triplets):
            prompt_value = _get_value(triplet, "prompt", {})
            response_value = _get_value(triplet, "response", {})
            prompt_ids = list(_token_ids(prompt_value))
            response_ids = list(_token_ids(response_value))
            if not prompt_ids or not response_ids:
                previous_trace = prompt_ids + response_ids
                continue

            if previous_trace and not ids_startswith(prompt_ids, previous_trace):
                if len(mismatch_rows) < _TRACE_MERGE_MISMATCH_WANDB_LIMIT:
                    mismatch_rows.append(
                        {
                            "global_steps": global_steps,
                            "rollout_id": rollout_id,
                            "data_id": _get_value(rollout, "data_id", None),
                            "turn_index": turn_index,
                            "template_mismatch": True,
                            "retoken_mismatch": False,
                            "others_mismatch": False,
                            "prompt_length": len(prompt_ids),
                            "response_length": len(response_ids),
                            "previous_trace_length": len(previous_trace),
                            "current_trace_length": len(prompt_ids),
                            "previous_trace": _decode_trace_text(tokenizer, previous_trace),
                            "current_trace": _decode_trace_text(tokenizer, prompt_ids),
                        }
                    )

            rows.append(
                {
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "reward": float(reward),
                    "rollout_id": rollout_id,
                    "data_id": _get_value(rollout, "data_id", None),
                    "image_urls": _get_image_urls(prompt_value),
                }
            )
            previous_trace = prompt_ids + response_ids

    if global_steps is not None and mismatch_rows:
        _upload_trace_merge_mismatches_to_wandb(mismatch_rows, global_steps)

    if global_steps is not None:
        records = _build_compact_rollout_trajectory_records(
            rollouts,
            tokenizer=tokenizer,
            reward_fillna_value=reward_fillna_value,
            limit=_ROLLOUT_TRAJECTORY_WANDB_LIMIT,
        )
        if records:
            _upload_compact_rollout_trajectories_to_wandb(
                records,
                global_steps,
                is_validation=is_validation,
            )

    if not rows:
        prompt_width = max(1, max_prompt_length)
        response_width = max(1, max_response_length)
        empty_batch = {
            "prompts": torch.empty((0, prompt_width), dtype=torch.long),
            "responses": torch.empty((0, response_width), dtype=torch.long),
            "input_ids": torch.empty((0, prompt_width + response_width), dtype=torch.long),
            "attention_mask": torch.empty((0, prompt_width + response_width), dtype=torch.long),
            "position_ids": torch.empty((0, prompt_width + response_width), dtype=torch.long),
        }
        return _make_dataproto(empty_batch, {"reward": np.asarray([], dtype=np.float32)})

    prompts: list[list[int]] = []
    prompt_masks: list[list[int]] = []
    responses: list[list[int]] = []
    response_masks: list[list[int]] = []

    for row in rows:
        prompt, prompt_mask = get_left_padded_ids_and_attention_mask(
            row["prompt_ids"], max_prompt_length, int(pad_token_id)
        )
        response, response_mask = get_right_padded_ids_and_attention_mask(
            row["response_ids"], max_response_length, int(pad_token_id)
        )
        prompts.append(prompt)
        prompt_masks.append(prompt_mask)
        responses.append(response)
        response_masks.append(response_mask)

    prompt_tensor = torch.tensor(prompts, dtype=torch.long)
    response_tensor = torch.tensor(responses, dtype=torch.long)
    prompt_mask_tensor = torch.tensor(prompt_masks, dtype=torch.long)
    response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)
    input_ids = torch.cat((prompt_tensor, response_tensor), dim=1)
    attention_mask = torch.cat((prompt_mask_tensor, response_mask_tensor), dim=1)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1)

    batch: dict[str, torch.Tensor] = {
        "prompts": prompt_tensor,
        "responses": response_tensor,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    multi_modal_inputs: list[Any] = []
    has_multimodal = False
    for index, row in enumerate(rows):
        current_inputs = _build_multi_modal_inputs(processor, row["image_urls"])
        multi_modal_inputs.append(current_inputs)
        if current_inputs is not None:
            has_multimodal = True
            mrope_ids = _compute_mrope_position_ids(
                processor,
                input_ids[index : index + 1],
                attention_mask[index : index + 1],
                current_inputs,
            )
            if mrope_ids is not None:
                if "mrope_position_ids" not in batch:
                    batch["mrope_position_ids"] = torch.zeros(
                        (len(rows), *mrope_ids.shape[1:]), dtype=mrope_ids.dtype
                    )
                batch["mrope_position_ids"][index] = mrope_ids[0]

    non_tensor_batch: dict[str, Any] = {
        "reward": np.asarray([row["reward"] for row in rows], dtype=np.float32),
        "rollout_id": np.asarray([row["rollout_id"] for row in rows], dtype=object),
        "data_id": np.asarray([row["data_id"] for row in rows], dtype=object),
    }
    if has_multimodal:
        non_tensor_batch["multi_modal_inputs"] = np.asarray(multi_modal_inputs, dtype=object)

    return _make_dataproto(batch, non_tensor_batch)


def rollouts_to_data_proto(
    rollouts: list[CompletedRollout],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int | None = None,
    reward_fillna_value: float = 0.0,
    processor: Any | None = None,
    global_steps: int | None = None,
    is_validation: bool = False,
) -> DataProto:
    return convert_rollouts_to_data_proto(
        rollouts,
        tokenizer,
        max_prompt_length,
        max_response_length,
        pad_token_id,
        reward_fillna_value,
        processor,
        global_steps,
        is_validation,
    )