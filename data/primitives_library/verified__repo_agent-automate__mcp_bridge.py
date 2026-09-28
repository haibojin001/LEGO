from __future__ import annotations

import json
from typing import Any


def _build_mcp(state):
    """Construct a FastMCP instance with all autoMate tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit(
            "Install with: pip install 'autoMate[mcp]' (mcp[cli] required)"
        ) from e

    mcp = FastMCP("autoMate", json_response=True, streamable_http_path="/")

    @mcp.tool()
    def automate(prompt: str) -> str:
        """Run an autoMate agent loop from a natural-language prompt.

        autoMate plans, picks tools, fills in parameters, executes, and returns
        the final answer. Best when the caller wants autoMate to figure out the
        steps. For specific operations (look up a note, read a file), the
        caller should pick the dedicated tool below directly.
        """
        try:
            result = state.agent.run(prompt, source="mcp")
        except RuntimeError as e:
            return f"autoMate error: {e}"
        return result.final or "(no final answer)"

    for tool in state.registry.all():
        _wrap_tool_for_mcp(mcp, tool)

    return mcp


def serve_stdio() -> None:
    from .state import build_state

    state = build_state()
    mcp = _build_mcp(state)
    mcp.run()


def build_http_app(state):
    """Return a Starlette ASGI app that speaks MCP over HTTP/SSE."""
    mcp = _build_mcp(state)
    return mcp.streamable_http_app(), mcp.session_manager


def _wrap_tool_for_mcp(mcp, tool) -> None:
    """Register a registry Tool with FastMCP using a generic kwargs handler."""

    def handler(**kwargs: Any) -> str:
        try:
            result = tool.call(kwargs)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

        if isinstance(result, str):
            return result

        return json.dumps(result, ensure_ascii=False, default=str)

    handler.__name__ = tool.name.replace(".", "_").replace("-", "_")
    handler.__doc__ = tool.description
    mcp.tool()(handler)