from typing import Optional

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


def get_rope_index(
    processor: "Qwen3VLProcessor",
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw,
            video_grid_thw[:, 0],
            dim=0,
        )
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        attention_mask = attention_mask.to(input_ids.device)
        unmasked_ids = input_ids[attention_mask == 1]
        vision_starts = torch.argwhere(unmasked_ids == vision_start_token_id)
        vision_payloads = unmasked_ids[vision_starts + 1]

        remaining_images = (vision_payloads == image_token_id).sum()
        remaining_videos = (vision_payloads == video_token_id).sum()

        token_values = unmasked_ids.tolist()
        segments = []
        offset = 0
        next_image_grid = 0
        next_video_grid = 0

        for _ in range(remaining_images + remaining_videos):
            if remaining_images > 0 and image_token_id in token_values:
                image_location = token_values.index(image_token_id, offset)
            else:
                image_location = len(token_values) + 1

            if remaining_videos > 0 and video_token_id in token_values:
                video_location = token_values.index(video_token_id, offset)
            else:
                video_location = len(token_values) + 1

            if image_location < video_location:
                grid_t, grid_h, grid_w = image_grid_thw[next_image_grid]
                next_image_grid += 1
                remaining_images -= 1
                visual_location = image_location
            else:
                grid_t, grid_h, grid_w = video_grid_thw[next_video_grid]
                next_video_grid += 1
                remaining_videos -= 1
                visual_location = video_location

            llm_t = grid_t.item()
            llm_h = grid_h.item() // merge_size
            llm_w = grid_w.item() // merge_size

            text_count = visual_location - offset
            segment_start = segments[-1].max() + 1 if segments else 0

            text_positions = torch.arange(text_count).view(1, -1).expand(3, -1)
            segments.append(text_positions + segment_start)

            temporal_positions = (
                torch.arange(llm_t)
                .view(-1, 1)
                .expand(-1, llm_h * llm_w)
                .flatten()
            )
            vertical_positions = (
                torch.arange(llm_h)
                .view(1, -1, 1)
                .expand(llm_t, -1, llm_w)
                .flatten()
            )
            horizontal_positions = (
                torch.arange(llm_w)
                .view(1, 1, -1)
                .expand(llm_t, llm_h, -1)
                .flatten()
            )

            visual_positions = torch.stack(
                (temporal_positions, vertical_positions, horizontal_positions)
            )
            segments.append(visual_positions + text_count + segment_start)

            offset = visual_location + llm_t * llm_h * llm_w

        if offset < len(token_values):
            segment_start = segments[-1].max() + 1 if segments else 0
            trailing_count = len(token_values) - offset
            trailing_positions = torch.arange(trailing_count).view(1, -1).expand(3, -1)
            segments.append(trailing_positions + segment_start)

        calculated_positions = torch.cat(segments, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = calculated_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
            )

    return position_ids


def _get_input_embeds(
    model: "Qwen3VLModel",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    embeddings = model.get_input_embeddings()(input_ids)
    image_mask = None
    video_mask = None
    image_deepstack = None
    video_deepstack = None

    if pixel_values is not None:
        image_pixels = pixel_values.type(model.visual.dtype)
        image_features, image_deepstack = model.visual(
            image_pixels,
            grid_thw=image_grid_thw,
        )

        image_token_count = (input_ids == model.config.image_token_id).sum().item()
        if image_token_count != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {image_token_count}, features {image_features.shape[0]}"
            )

        selected = input_ids == model.config.image_token_id
        image_mask = selected.unsqueeze(-1).expand_as(embeddings).to(embeddings.device)
        image_features = image_features.to(embeddings.device, embeddings.dtype)
        embeddings = embeddings.masked_scatter(image_mask, image_features)

    if pixel_values_videos is not None:
        video_pixels = pixel_values_videos.type(model.visual.dtype)
        video_features, video_deepstack = model.visual(
            video_pixels,
            grid_thw=video_grid_thw,
        )

        video_token_count = (input_ids == model.config.video_token_id).sum().item()
        if video_token_count != video_features.shape[0]:
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {video_token_count}, features {video_features.shape[0]}"
            )

        selected = input_ids == model.config.video_token_id
        video_mask = selected.unsqueeze(-1).expand_as(embeddings).to(embeddings.device)
        video_features = video_features.to(embeddings.device, embeddings.dtype)
        embeddings = embeddings.masked_scatter(video_mask, video_features)

    visual_pos_masks = None
    deepstack_visual_embeds = None

    if image_mask is not None and video_mask is not None:
        image_positions = image_mask[..., 0]
        video_positions = video_mask[..., 0]
        visual_pos_masks = image_positions | video_positions

        joined_image_positions = image_positions[visual_pos_masks]
        joined_video_positions = video_positions[visual_pos_masks]
        deepstack_visual_embeds = []

        for image_layer, video_layer in zip(image_deepstack, video_deepstack):
            image_layer = image_layer.to(embeddings.device, embeddings.dtype)
            video_layer = video_layer.to(embeddings.device, embeddings.dtype)
            merged_layer = image_layer.new_zeros(
                (visual_pos_masks.sum(), image_layer.shape[-1])
            )
            merged_layer[joined_image_positions] = image_layer
            merged_layer[joined_video_positions] = video_layer
            deepstack_visual_embeds.append(merged_layer)
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = [
            layer.to(embeddings.device, embeddings.dtype) for layer in image_deepstack
        ]
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        deepstack_visual_embeds = [
            layer.to(embeddings.device, embeddings.dtype) for layer in video_deepstack
        ]

    return embeddings, visual_pos_masks, deepstack_visual_embeds