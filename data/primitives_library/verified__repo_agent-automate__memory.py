from __future__ import annotations

from .. import memory as _memory
from ..store import get_db as _get_db
from .registry import Tool, ToolRegistry


def register(reg: ToolRegistry) -> None:
    def store_fact(key, value):
        _memory.set_value(_get_db(), key, value)
        return {"ok": True}

    def fetch_fact(key):
        return {"value": _memory.get_value(_get_db(), key)}

    def list_facts(prefix=""):
        return {"items": _memory.list_all(_get_db(), prefix=prefix)}

    def forget_fact(key):
        return {"deleted": _memory.delete(_get_db(), key)}

    reg.register(
        Tool(
            name="memory.set",
            description=(
                "Store a fact for future sessions. Use dotted keys like "
                "'user.preferred_language' or 'project.alpha.repo'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
            handler=store_fact,
            category="memory",
        )
    )
    reg.register(
        Tool(
            name="memory.get",
            description="Retrieve a previously-stored fact by key.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            handler=fetch_fact,
            category="memory",
        )
    )
    reg.register(
        Tool(
            name="memory.list",
            description="List all stored facts. Pass a prefix to filter.",
            parameters={
                "type": "object",
                "properties": {"prefix": {"type": "string", "default": ""}},
            },
            handler=list_facts,
            category="memory",
        )
    )
    reg.register(
        Tool(
            name="memory.delete",
            description="Forget a fact.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            handler=forget_fact,
            category="memory",
            danger="medium",
        )
    )