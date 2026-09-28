from __future__ import annotations

import json
import urllib.request
from typing import Iterator

from .base import ChatMessage, ChatResponse, ProviderClient, ToolCall, ToolSpec


class AnthropicClient(ProviderClient):
    spec_id = "anthropic"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_model: str = "",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        system_parts = [message.content for message in messages if message.role == "system"]
        converted: list[dict] = []

        for message in messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                        ],
                    }
                )
                continue

            content: list[dict] = []
            if message.content:
                content.append({"type": "text", "text": message.content})

            for tool_call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    }
                )

            converted.append(
                {
                    "role": message.role,
                    "content": content or [{"type": "text", "text": ""}],
                }
            )

        return "\n\n".join(system_parts), converted

    def chat(self, messages, *, model, tools=None, temperature=0.2, max_tokens=None):
        system, converted_messages = self._split_system(messages)
        payload: dict = {
            "model": model or self.default_model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters
                    or {"type": "object", "properties": {}},
                }
                for tool in tools
            ]

        request = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode())

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in result.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        return ChatResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            raw=result,
        )

    def stream(self, *args, **kwargs) -> Iterator[dict]:
        raise NotImplementedError("Streaming not implemented for Anthropic adapter")