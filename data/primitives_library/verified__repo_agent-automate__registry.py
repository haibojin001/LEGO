from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    category: str = "general"
    requires: list[str] = field(default_factory=list)
    danger: str = "low"
    tier: str = "free"

    def call(self, args: dict[str, Any]) -> Any:
        signature = inspect.signature(self.handler)
        supplied = {
            argument_name: argument_value
            for argument_name, argument_value in args.items()
            if argument_name in signature.parameters
        }
        return self.handler(**supplied)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def add(
        self,
        *,
        name: str,
        description: str,
        parameters: dict | None = None,
        category: str = "general",
        requires: list[str] | None = None,
        danger: str = "low",
    ) -> Callable[[Callable], Callable]:
        def decorate(function: Callable) -> Callable:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters or {
                        "type": "object",
                        "properties": {},
                    },
                    handler=function,
                    category=category,
                    requires=requires or [],
                    danger=danger,
                )
            )
            return function

        return decorate

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def by_category(self) -> dict[str, list[Tool]]:
        categories: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            categories.setdefault(tool.category, []).append(tool)
        return categories


def build_default_registry() -> ToolRegistry:
    from . import audio as audio_tools
    from . import browser as browser_tools
    from . import browser_ext as browser_extension_tools
    from . import desktop as desktop_tools
    from . import files as file_tools
    from . import integrations_adapter as integration_tools
    from . import memory as memory_tools
    from . import notes as note_tools
    from . import reminders as reminder_tools
    from . import script as script_tools
    from . import search as search_tools
    from . import shell as shell_tools

    registry = ToolRegistry()

    search_tools.register(registry)
    note_tools.register(registry)
    file_tools.register(registry)
    reminder_tools.register(registry)
    memory_tools.register(registry)
    audio_tools.register(registry)

    shell_tools.register(registry)
    script_tools.register(registry)
    browser_tools.register(registry)
    browser_extension_tools.register(registry)
    desktop_tools.register(registry)
    integration_tools.register(registry)

    return registry