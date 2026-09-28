from typing import Optional

import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
)
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


def get_rope_index(
    processor: "Qwen2VLProcessor",
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    merge_size = processor.image_processor.merge_size
    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3,
            input_ids.size(0),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        unmasked_ids = input_ids[attention_mask == 1]
        starts = torch.argwhere(unmasked_ids == vision_start_token)
        media_tokens = unmasked_ids[starts + 1]

        image_count = (media_tokens == image_token).sum()
        video_count = (media_tokens == video_token).sum()

        ids_as_list = unmasked_ids.tolist()
        position_pieces = []
        start = 0
        image_number = 0
        video_number = 0
        images_left = image_count
        videos_left = video_count

        for _ in range(image_count + video_count):
            if image_token in ids_as_list and images_left > 0:
                image_end = ids_as_list.index(image_token, start)
            else:
                image_end = len(ids_as_list) + 1

            if video_token in ids_as_list and videos_left > 0:
                video_end = ids_as_list.index(video_token, start)
            else:
                video_end = len(ids_as_list) + 1

            if image_end < video_end:
                temporal, height, width = image_grid_thw[image_number]
                temporal_seconds = 0
                image_number += 1
                images_left -= 1
                end = image_end
            else:
                temporal, height, width = video_grid_thw[video_number]
                temporal_seconds = (
                    second_per_grid_ts[video_number]
                    if second_per_grid_ts is not None
                    else 1.0
                )
                video_number += 1
                videos_left -= 1
                end = video_end

            temporal = temporal.item()
            height = height.item() // merge_size
            width = width.item() // merge_size
            text_length = end - start

            offset = position_pieces[-1].max() + 1 if position_pieces else 0
            position_pieces.append(
                torch.arange(text_length).view(1, -1).expand(3, -1) + offset
            )

            temporal_indices = torch.arange(temporal).view(-1, 1).expand(
                -1, height * width
            )
            temporal_indices = (temporal_indices * temporal_seconds * 2).long().flatten()
            height_indices = torch.arange(height).view(1, -1, 1).expand(
                temporal, -1, width
            ).flatten()
            width_indices = torch.arange(width).view(1, 1, -1).expand(
                temporal, height, -1
            ).flatten()

            position_pieces.append(
                torch.stack((temporal_indices, height_indices, width_indices))
                + text_length
                + offset
            )
            start = end + temporal * height * width

        if start < len(ids_as_list):
            offset = position_pieces[-1].max() + 1 if position_pieces else 0
            trailing_length = len(ids_as_list) - start
            position_pieces.append(
                torch.arange(trailing_length).view(1, -1).expand(3, -1) + offset
            )

        positions = torch.cat(position_pieces, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
            )

    return position_ids


def _get_input_embeds(
    model: "Qwen2VLModel",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_features = model.visual(pixel_values, grid_thw=image_grid_thw)

        image_token_count = (input_ids == model.config.image_token_id).sum().item()
        image_feature_count = image_features.shape[0]
        if image_token_count != image_feature_count:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {image_token_count}, features {image_feature_count}"
            )

        image_locations = input_ids == model.config.image_token_id
        image_locations = image_locations.unsqueeze(-1).expand_as(inputs_embeds)
        image_locations = image_locations.to(inputs_embeds.device)

        image_features = image_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_locations, image_features)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_features = model.visual(pixel_values_videos, grid_thw=video_grid_thw)

        video_token_count = (input_ids == model.config.video_token_id).sum().item()
        video_feature_count = video_features.shape[0]
        if video_token_count != video_feature_count:
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {video_token_count}, features {video_feature_count}"
            )

        video_locations = input_ids == model.config.video_token_id
        video_locations = video_locations.unsqueeze(-1).expand_as(inputs_embeds)
        video_locations = video_locations.to(inputs_embeds.device)

        video_features = video_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_locations, video_features)

    if pixel_values is None and pixel_values_videos is None:
        vision_config = model.config.vision_config
        patch_dimension = (
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * vision_config.patch_size**2
        )
        placeholder_pixels = torch.zeros(
            (16, patch_dimension),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        placeholder_grid = torch.tensor(
            [[1, 4, 4]],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        placeholder_features = model.visual(
            placeholder_pixels,
            grid_thw=placeholder_grid,
        )
        inputs_embeds += placeholder_features.mean() * 0.0

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }