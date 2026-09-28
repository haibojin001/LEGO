from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

from .base import ChatMessage, ChatResponse, ProviderClient, ToolCall, ToolSpec


class OpenAICompatClient(ProviderClient):
    def __init__(
        self,
        *,
        spec_id: str,
        base_url: str,
        api_key: str | None,
        default_model: str | None = None,
        timeout: int = 120,
    ):
        self.spec_id = spec_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.default_model = default_model or ""
        self.timeout = timeout

    def _headers(self) -> dict:
        result = {"Content-Type": "application/json"}
        if self.api_key:
            result["Authorization"] = f"Bearer {self.api_key}"
        return result

    @staticmethod
    def _serialize_messages(messages: list[ChatMessage]) -> list[dict]:
        result: list[dict] = []
        for message in messages:
            data: dict = {
                "role": message.role,
                "content": message.content or "",
            }
            if message.name:
                data["name"] = message.name
            if message.tool_calls:
                data["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.tool_call_id:
                data["tool_call_id"] = message.tool_call_id
            result.append(data)
        return result

    def chat(self, messages, *, model, tools=None, temperature=0.2, max_tokens=None):
        payload = {
            "model": model or self.default_model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [tool.to_openai() for tool in tools]
            payload["tool_choice"] = "auto"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"{self.spec_id} HTTP {error.code}: {error.read().decode()[:500]}"
            ) from error

        message = result["choices"][0]["message"]
        calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments_text = function.get("arguments") or "{}"
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError:
                arguments = {"_raw": function.get("arguments")}
            calls.append(
                ToolCall(
                    id=call["id"],
                    name=function["name"],
                    arguments=arguments,
                )
            )

        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            raw=result,
        )

    def stream(
        self,
        messages,
        *,
        model,
        tools=None,
        temperature=0.2,
        max_tokens=None,
    ) -> Iterator[dict]:
        payload = {
            "model": model or self.default_model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [tool.to_openai() for tool in tools]

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for line in response:
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {})
                yield {
                    "delta": delta.get("content") or "",
                    "tool_calls": delta.get("tool_calls") or [],
                }