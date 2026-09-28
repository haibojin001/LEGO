from __future__ import annotations

import json
from typing import Any, Iterator

from .base import ChatMessage, ChatResponse, ProviderClient, ToolCall, ToolSpec


class LiteLLMClient(ProviderClient):
    spec_id = "litellm"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

    @staticmethod
    def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for message in messages:
            payload: dict[str, Any] = {
                "role": message.role,
                "content": message.content or "",
            }

            if message.name:
                payload["name"] = message.name

            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            if message.tool_call_id:
                payload["tool_call_id"] = message.tool_call_id

            result.append(payload)

        return result

    def _build_kwargs(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
            "drop_params": True,
            "timeout": self.timeout,
        }

        if max_tokens:
            options["max_tokens"] = max_tokens

        if self.api_key:
            options["api_key"] = self.api_key

        if self.api_base:
            options["api_base"] = self.api_base

        if tools:
            options["tools"] = [tool.to_openai() for tool in tools]
            options["tool_choice"] = "auto"

        return options

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        import litellm

        options = self._build_kwargs(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = litellm.completion(**options)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[dict]:
        import litellm

        options = self._build_kwargs(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        options["stream"] = True

        for chunk in litellm.completion(**options):
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                yield {
                    "delta": getattr(delta, "content", "") or "",
                    "tool_calls": getattr(delta, "tool_calls", []) or [],
                }

    @staticmethod
    def _parse_response(response: Any) -> ChatResponse:
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
        tool_calls: list[ToolCall] = []

        for tool_call in getattr(message, "tool_calls", None) or []:
            function = tool_call.function
            try:
                arguments = json.loads(function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": function.arguments}

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=function.name,
                    arguments=arguments,
                )
            )

        return ChatResponse(content=content, tool_calls=tool_calls, raw=response)