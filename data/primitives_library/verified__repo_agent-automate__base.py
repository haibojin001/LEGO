from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod


class BaseIntegration(ABC):
    name: str
    label: str
    env_vars: dict[str, str]

    def is_configured(self) -> bool:
        return all(os.environ.get(key) for key in self.env_vars)

    def env(self, key: str) -> str:
        return os.environ.get(key, "")

    @abstractmethod
    def register(self, mcp) -> None:
        """Register MCP tools onto the FastMCP server."""
        ...

    def config_hint(self) -> str:
        missing = [
            f"  {key} — {description}"
            for key, description in self.env_vars.items()
            if not os.environ.get(key)
        ]
        return f"{self.label} not configured. Set:\n" + "\n".join(missing)

    @staticmethod
    def get(url: str, headers: dict | None = None) -> dict:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())

    @staticmethod
    def post(url: str, data: dict, headers: dict | None = None) -> dict:
        body = json.dumps(data).encode()
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        request = urllib.request.Request(url, data=body, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return {"error": error.code, "msg": error.read().decode()}

    @staticmethod
    def patch(url: str, data: dict, headers: dict | None = None) -> dict:
        body = json.dumps(data).encode()
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="PATCH",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return {"error": error.code, "msg": error.read().decode()}

    @staticmethod
    def put(url: str, data: dict, headers: dict | None = None) -> dict:
        body = json.dumps(data).encode()
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return {"error": error.code, "msg": error.read().decode()}

    @staticmethod
    def ok(result: dict) -> str:
        return json.dumps(result, ensure_ascii=False, indent=2)