from __future__ import annotations

import threading
from typing import Any

from .registry import Tool, ToolRegistry


_lock = threading.Lock()
_state: dict[str, Any] = {
    "playwright": None,
    "browser": None,
    "context": None,
    "page": None,
}


def _ensure_page(headless: bool = False):
    with _lock:
        existing = _state["page"]
        if existing:
            return existing

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install 'autoMate[browser]' "
                "and then 'python -m playwright install chromium'."
            ) from exc

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        _state.update(
            {
                "playwright": playwright,
                "browser": browser,
                "context": context,
                "page": page,
            }
        )
        return page


def _close():
    with _lock:
        for name in ("page", "context", "browser"):
            resource = _state.get(name)
            if resource:
                try:
                    resource.close()
                except Exception:
                    pass

        playwright = _state.get("playwright")
        if playwright:
            try:
                playwright.stop()
            except Exception:
                pass

        for name in ("page", "context", "browser", "playwright"):
            _state[name] = None


def register(reg: ToolRegistry) -> None:
    def open_url(url: str, headless: bool = False) -> dict:
        _ensure_page(headless)
        page = _state["page"]
        page.goto(url, wait_until="domcontentloaded")
        return {"url": page.url, "title": page.title()}

    reg.register(
        Tool(
            name="browser.open",
            description="Open a URL in a managed Chromium tab. Reuses one page across calls.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "headless": {"type": "boolean", "default": False},
                },
                "required": ["url"],
            },
            handler=open_url,
            category="browser",
            danger="medium",
        )
    )

    def _click(selector: str) -> dict:
        page = _ensure_page()
        page.locator(selector).first.click()
        return {"clicked": selector, "url": page.url}

    reg.register(
        Tool(
            name="browser.click",
            description="Click the first element matching a CSS selector or Playwright text= locator.",
            parameters={
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
            handler=_click,
            category="browser",
            danger="medium",
        )
    )

    def _type(selector: str, text: str, submit: bool = False) -> dict:
        page = _ensure_page()
        locator = page.locator(selector).first
        locator.fill(text)
        if submit:
            locator.press("Enter")
        return {"typed_into": selector, "url": page.url}

    reg.register(
        Tool(
            name="browser.type",
            description="Fill a form field. Set submit=true to press Enter after typing.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                },
                "required": ["selector", "text"],
            },
            handler=_type,
            category="browser",
            danger="medium",
        )
    )

    def _extract(selector: str | None = None, kind: str = "text") -> dict:
        page = _ensure_page()
        if not selector:
            return {
                "url": page.url,
                "title": page.title(),
                "text": page.locator("body").inner_text()[:8000],
            }

        locator = page.locator(selector)
        if kind == "html":
            return {
                "items": [
                    element.inner_html() for element in locator.element_handles()
                ][:50]
            }
        if kind == "attribute":
            return {
                "items": [
                    element.get_attribute("href") or element.get_attribute("src")
                    for element in locator.element_handles()
                ][:50]
            }
        return {
            "items": [
                element.inner_text() for element in locator.element_handles()
            ][:50]
        }

    reg.register(
        Tool(
            name="browser.extract",
            description=(
                "Pull text/html/links from the current page. Pass no selector to grab the "
                "whole body text (truncated to 8KB)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["text", "html", "attribute"],
                        "default": "text",
                    },
                },
            },
            handler=_extract,
            category="browser",
        )
    )

    def _screenshot(full_page: bool = False) -> dict:
        import base64

        page = _ensure_page()
        png = page.screenshot(full_page=full_page)
        return {"format": "png", "base64": base64.b64encode(png).decode()}

    reg.register(
        Tool(
            name="browser.screenshot",
            description="Capture the current page (PNG, base64-encoded).",
            parameters={
                "type": "object",
                "properties": {
                    "full_page": {"type": "boolean", "default": False},
                },
            },
            handler=_screenshot,
            category="browser",
        )
    )

    reg.register(
        Tool(
            name="browser.close",
            description="Close the managed browser and free its resources.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: (_close(), {"closed": True})[1],
            category="browser",
        )
    )