from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class _Pending:
    event: threading.Event
    result: dict | None = None

    def complete(self, msg: dict) -> None:
        self.result = msg
        self.event.set()


class ExtensionBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outgoing: asyncio.Queue | None = None
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self._connected_at: float | None = None
        self._client_info: dict = {}

    def attach(
        self,
        loop: asyncio.AbstractEventLoop,
        client_info: dict | None = None,
    ) -> None:
        self._loop = loop
        self._outgoing = asyncio.Queue()
        self._connected_at = time.time()
        self._client_info = client_info or {}

    def detach(self) -> None:
        self._connected_at = None
        with self._lock:
            for request in self._pending.values():
                request.complete(
                    {"ok": False, "error": "extension disconnected"}
                )
            self._pending.clear()
        self._outgoing = None
        self._loop = None
        self._client_info = {}

    @property
    def connected(self) -> bool:
        return self._connected_at is not None

    @property
    def status(self) -> dict:
        return {
            "connected": self.connected,
            "connected_at": self._connected_at,
            "client": self._client_info,
            "pending": len(self._pending),
        }

    async def next_outgoing(self) -> str:
        assert self._outgoing is not None
        return await self._outgoing.get()

    def deliver(self, msg: dict) -> None:
        request_id = msg.get("id")
        with self._lock:
            request = self._pending.pop(request_id, None)
        if request is not None:
            request.complete(msg)

    def call(
        self,
        cmd: str,
        args: dict | None = None,
        *,
        timeout: float = 30.0,
    ) -> Any:
        if (
            not self.connected
            or self._loop is None
            or self._outgoing is None
        ):
            raise RuntimeError(
                "No browser extension connected. Install the autoMate extension "
                "from chrome://extensions (Load unpacked → ./extension/)."
            )

        request_id = uuid.uuid4().hex
        request = _Pending(event=threading.Event())

        with self._lock:
            self._pending[request_id] = request

        message = json.dumps(
            {
                "id": request_id,
                "cmd": cmd,
                "args": args or {},
            }
        )
        asyncio.run_coroutine_threadsafe(
            self._outgoing.put(message),
            self._loop,
        )

        if not request.event.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(
                f"extension call '{cmd}' timed out after {timeout}s"
            )

        response = request.result or {}
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "extension error"))
        return response.get("result")


bus = ExtensionBus()