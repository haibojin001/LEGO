from __future__ import annotations

import base64
import io
from typing import Any

from .registry import Tool, ToolRegistry


def _has_display() -> bool:
    try:
        import pyautogui
    except Exception:
        return False
    return True


def _screenshot(region: list[int] | None = None) -> dict:
    import pyautogui

    screenshot = pyautogui.screenshot(
        region=tuple(region) if region else None
    )
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")

    return {
        "format": "png",
        "base64": base64.b64encode(buffer.getvalue()).decode(),
        "width": screenshot.width,
        "height": screenshot.height,
    }


def _click(x: int, y: int, button: str = "left") -> dict:
    import pyautogui

    pyautogui.click(x=x, y=y, button=button)
    return {"clicked": [x, y], "button": button}


def _type(text: str, interval: float = 0.02) -> dict:
    import pyautogui

    pyautogui.typewrite(text, interval=interval)
    return {"typed": text}


def _press(key: str) -> dict:
    import pyautogui

    pyautogui.press(key)
    return {"pressed": key}


def _size() -> dict:
    import pyautogui

    width, height = pyautogui.size()
    return {"width": width, "height": height}


def register(reg: ToolRegistry) -> None:
    if not _has_display():
        return

    reg.register(
        Tool(
            name="desktop.screenshot",
            description="Capture the screen (full or a [x, y, w, h] region) as base64 PNG.",
            parameters={
                "type": "object",
                "properties": {
                    "region": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
            },
            handler=_screenshot,
            category="desktop",
        )
    )

    reg.register(
        Tool(
            name="desktop.click",
            description="Click at absolute screen coordinates.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "default": "left",
                    },
                },
                "required": ["x", "y"],
            },
            handler=_click,
            category="desktop",
            danger="medium",
        )
    )

    reg.register(
        Tool(
            name="desktop.type",
            description="Type a string at the focused element.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "interval": {"type": "number", "default": 0.02},
                },
                "required": ["text"],
            },
            handler=_type,
            category="desktop",
            danger="medium",
        )
    )

    reg.register(
        Tool(
            name="desktop.press",
            description="Press a single key (e.g. 'enter', 'tab', 'esc', 'cmd').",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
            handler=_press,
            category="desktop",
            danger="medium",
        )
    )

    reg.register(
        Tool(
            name="desktop.size",
            description="Return the primary screen size in pixels.",
            parameters={"type": "object", "properties": {}},
            handler=_size,
            category="desktop",
        )
    )