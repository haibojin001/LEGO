from __future__ import annotations

from ..extension_bus import bus
from .registry import Tool, ToolRegistry


def _call(cmd: str, args: dict | None = None, *, timeout: float = 30.0):
    return bus.call(cmd, args or {}, timeout=timeout)


def register(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="bx.tabs",
            description="List open tabs in the user's browser. Returns id, title, url, active flag.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: _call("tabs.list"),
            category="browser-ext",
        )
    )
    reg.register(
        Tool(
            name="bx.open",
            description="Open a new tab pointed at a URL. Returns the new tab id.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "active": {"type": "boolean", "default": True},
                },
                "required": ["url"],
            },
            handler=lambda url, active=True: _call(
                "tabs.open", {"url": url, "active": active}
            ),
            category="browser-ext",
            danger="medium",
        )
    )
    reg.register(
        Tool(
            name="bx.activate",
            description="Switch focus to the tab with the given id.",
            parameters={
                "type": "object",
                "properties": {"tab_id": {"type": "integer"}},
                "required": ["tab_id"],
            },
            handler=lambda tab_id: _call("tabs.activate", {"tab_id": tab_id}),
            category="browser-ext",
        )
    )
    reg.register(
        Tool(
            name="bx.close",
            description="Close a tab by id.",
            parameters={
                "type": "object",
                "properties": {"tab_id": {"type": "integer"}},
                "required": ["tab_id"],
            },
            handler=lambda tab_id: _call("tabs.close", {"tab_id": tab_id}),
            category="browser-ext",
            danger="medium",
        )
    )
    reg.register(
        Tool(
            name="bx.navigate",
            description="Navigate the active tab (or a specific tab_id) to a URL.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "tab_id": {"type": "integer"},
                },
                "required": ["url"],
            },
            handler=lambda url, tab_id=None: _call(
                "tabs.navigate", {"url": url, "tab_id": tab_id}
            ),
            category="browser-ext",
            danger="medium",
        )
    )
    reg.register(
        Tool(
            name="bx.screenshot",
            description="Capture the visible area of the active tab as base64 PNG.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: _call("tabs.screenshot"),
            category="browser-ext",
        )
    )
    reg.register(
        Tool(
            name="bx.click",
            description="Click an element in the active tab. Selector is CSS or text= (Playwright-like).",
            parameters={
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
            handler=lambda selector: _call("dom.click", {"selector": selector}),
            category="browser-ext",
            danger="medium",
        )
    )
    reg.register(
        Tool(
            name="bx.type",
            description="Type text into a focused/specified element in the active tab. Set submit=true to press Enter.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                },
                "required": ["selector", "text"],
            },
            handler=lambda selector, text, submit=False: _call(
                "dom.type",
                {"selector": selector, "text": text, "submit": submit},
            ),
            category="browser-ext",
            danger="medium",
        )
    )
    reg.register(
        Tool(
            name="bx.extract",
            description=(
                "Extract data from the active tab. kind=text returns inner text; "
                "html returns outer HTML; links returns hrefs; meta returns title+url+description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["text", "html", "links", "meta"],
                        "default": "text",
                    },
                    "limit": {"type": "integer", "default": 50},
                },
            },
            handler=lambda selector="", kind="text", limit=50: _call(
                "dom.extract",
                {"selector": selector, "kind": kind, "limit": limit},
            ),
            category="browser-ext",
        )
    )
    reg.register(
        Tool(
            name="bx.scroll",
            description="Scroll the active tab. direction = 'up' | 'down' | 'top' | 'bottom'.",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "top", "bottom"],
                        "default": "down",
                    },
                    "amount": {"type": "integer", "default": 600},
                },
            },
            handler=lambda direction="down", amount=600: _call(
                "dom.scroll",
                {"direction": direction, "amount": amount},
            ),
            category="browser-ext",
        )
    )
    reg.register(
        Tool(
            name="bx.eval",
            description=(
                "Run a JavaScript expression in the active tab and return its value. "
                "Powerful — only use when other bx.* tools won't do."
            ),
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=lambda expression: _call(
                "dom.eval",
                {"expression": expression},
            ),
            category="browser-ext",
            danger="high",
        )
    )