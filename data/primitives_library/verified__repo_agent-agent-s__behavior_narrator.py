from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import (
    call_llm_formatted,
    split_thinking_response,
    compress_image,
)
from gui_agents.s3.utils.formatters import THOUGHTS_ANSWER_TAG_FORMATTER
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Dict
import base64
import cv2
import numpy as np


class BehaviorNarrator:
    def __init__(self, engine_params):
        self.judge_agent = LMMAgent(engine_params=engine_params)

    @staticmethod
    def extract_mouse_action(action: str) -> list[str]:
        mouse_actions = []
        for sub_action in action.split(";"):
            sub_action = sub_action.strip()
            if (
                sub_action.startswith("pyautogui.click")
                or sub_action.startswith("pyautogui.moveTo")
                or sub_action.startswith("pyautogui.dragTo")
            ):
                mouse_actions.append(sub_action)
        return mouse_actions

    @staticmethod
    def mark_action(mouse_actions: list[str], img: Image):
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(25)

        drag_start_width, drag_start_height = None, None

        for mouse_action in mouse_actions:
            width, height = mouse_action.split("(")[1].strip(")").split(", ")[:2]
            width, height = int(width), int(height)

            width = max(0, min(img.width - 1, width))
            height = max(0, min(img.height - 1, height))

            def place_text(label, color, x, y):
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                offset_x, offset_y = -5, 5

                if x + offset_x + text_w > img.width:
                    offset_x = -text_w - 5
                if y + offset_y + text_h > img.height:
                    offset_y = -text_h - 5
                if x + offset_x < 0:
                    offset_x = 5
                if y + offset_y < 0:
                    offset_y = 5

                draw.text((x + offset_x, y + offset_y), label, fill=color, font=font)

            if mouse_action.startswith("pyautogui.click"):
                draw.circle((width, height), radius=3, fill=(255, 0, 0))
                place_text("Click", (255, 0, 0), width, height)

            if mouse_action.startswith("pyautogui.moveTo"):
                draw.circle((width, height), radius=3, fill=(0, 0, 255))
                place_text("MoveTo", (0, 0, 255), width, height)
                drag_start_height, drag_start_width = height, width

            if mouse_action.startswith("pyautogui.dragTo"):
                draw.line(
                    [(drag_start_width, drag_start_height), (width, height)],
                    fill=(0, 255, 0),
                    width=2,
                )
                draw.circle((width, height), radius=3, fill=(0, 255, 0))
                place_text("DragTo", (0, 255, 0), width, height)

    @staticmethod
    def get_mouse_action_representation(mouse_actions: list[str]) -> str:
        assert len(mouse_actions) <= 2, (
            f"Multiple mouse action types found: {mouse_actions}"
        )

        if len(mouse_actions) == 1:
            action = mouse_actions[0]
            if action.startswith("pyautogui.click"):
                return (
                    "The red circle labeled 'Click' marks the position where "
                    "the mouse was clicked."
                )
            elif action.startswith("pyautogui.moveTo"):
                return (
                    "The blue circle labeled 'MoveTo' marks the position where "
                    "the mouse was moved to."
                )
            raise ValueError(f"Unknown single action type: {action}")

        assert mouse_actions[0].startswith(
            "pyautogui.moveTo"
        ) and mouse_actions[1].startswith("pyautogui.dragTo")
        return (
            "The blue circle labeled 'MoveTo' marks the starting position of the mouse.\n"
            "The green circle labeled 'DragTo' marks the ending position.\n"
            "The green line illustrates the mouse's drag path."
        )

    @staticmethod
    def get_zoomed_image(
        image_bytes: bytes,
        x: int,
        y: int,
        width: int = 300,
        height: int = 300,
        upscaling: bool = False,
        scale: int = 4,
        add_bounding_box: bool = False,
    ) -> bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        cx, cy = x - width // 2, y - height // 2
        W, H = img.size

        left = min(max(cx, 0), W - width)
        top = min(max(cy, 0), H - height)
        right = left + width
        bottom = top + height

        zoomed_img = img.crop((left, top, right, bottom))

        if add_bounding_box:
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            original_with_box_bytes = compress_image(image=draw_img)
        else:
            original_with_box_bytes = image_bytes

        if upscaling:
            zoomed_img = cv2.cvtColor(np.array(zoomed_img), cv2.COLOR_RGB2BGR)
            zoomed_img = cv2.resize(
                zoomed_img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4,
            )
            zoomed_img = cv2.fastNlMeansDenoisingColored(
                zoomed_img,
                None,
                5,
                5,
                7,
                21,
            )
            zoomed_img = Image.fromarray(
                cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB)
            )

        zoomed_img_bytes = compress_image(image=zoomed_img)
        return zoomed_img_bytes, original_with_box_bytes

    def judge(
        self,
        screenshot_num: int,
        before_img_bytes: bytes,
        after_img_bytes: bytes,
        pyautogui_action: str,
    ) -> Dict[str, str]:
        if pyautogui_action == "DONE":
            return {
                "fact_thoughts": "The agent has indicated that it is done with the task.",
                "fact_answer": "The agent has indicated that it is done with the task.",
            }

        if pyautogui_action == "FAIL":
            return {
                "fact_thoughts": (
                    "The agent has indicated that it is impossible to proceed "
                    "further with the task."
                ),
                "fact_answer": (
                    "The agent has indicated that it is impossible to proceed "
                    "further with the task."
                ),
            }

        mouse_actions = BehaviorNarrator.extract_mouse_action(pyautogui_action)

        before_img = Image.open(BytesIO(before_img_bytes))
        BehaviorNarrator.mark_action(mouse_actions, before_img)

        out_buffer = BytesIO()
        before_img.save(out_buffer, format="PNG")
        marked_before_img_bytes = out_buffer.getvalue()

        marked_before_img_message = {
            "type": "image_url",
            "image_url": {
                "url": (
                    "data:image/png;base64,"
                    f"{base64.b64encode(marked_before_img_bytes).decode('utf-8')}"
                ),
                "detail": "high",
            },
        }

        if mouse_actions:
            coords = mouse_actions[-1].split("(")[1].strip(")").split(", ")
            x, y = int(coords[0]), int(coords[1])

            zoomed_after_img_bytes, marked_after_img_bytes = (
                BehaviorNarrator.get_zoomed_image(
                    image_bytes=after_img_bytes,
                    x=x,
                    y=y,
                    width=300,
                    height=300,
                    upscaling=True,
                    add_bounding_box=True,
                )
            )
        else:
            zoomed_after_img_bytes = after_img_bytes
            marked_after_img_bytes = after_img_bytes

        marked_after_img_message = {
            "type": "image_url",
            "image_url": {
                "url": (
                    "data:image/png;base64,"
                    f"{base64.b64encode(marked_after_img_bytes).decode('utf-8')}"
                ),
                "detail": "high",
            },
        }

        zoomed_after_img_message = {
            "type": "image_url",
            "image_url": {
                "url": (
                    "data:image/png;base64,"
                    f"{base64.b64encode(zoomed_after_img_bytes).decode('utf-8')}"
                ),
                "detail": "high",
            },
        }

        action_representation = (
            BehaviorNarrator.get_mouse_action_representation(mouse_actions)
            if mouse_actions
            else "No mouse action was identified in the provided action."
        )

        prompt_text = (
            f"{PROCEDURAL_MEMORY}\n\n"
            "You are given screenshots before and after an action was executed. "
            "Your task is to describe, factually and concisely, what changed as a "
            "result of the action. Focus only on visible evidence and do not infer "
            "unsupported intent or hidden state.\n\n"
            f"Screenshot number: {screenshot_num}\n"
            f"Executed action: {pyautogui_action}\n"
            f"{action_representation}\n\n"
            "The first image is the screenshot before the action with the mouse "
            "action annotated. The second image is the screenshot after the action "
            "with the relevant region marked. The third image is a zoomed view of "
            "that marked region in the after screenshot. Provide your reasoning in "
            "<thoughts> tags and the factual observation in <answer> tags."
        )

        prompt = [
            {"type": "text", "text": prompt_text},
            marked_before_img_message,
            marked_after_img_message,
            zoomed_after_img_message,
        ]

        response = call_llm_formatted(
            self.judge_agent,
            prompt,
            THOUGHTS_ANSWER_TAG_FORMATTER,
        )
        thoughts, answer = split_thinking_response(response)

        return {
            "fact_thoughts": thoughts,
            "fact_answer": answer,
        }