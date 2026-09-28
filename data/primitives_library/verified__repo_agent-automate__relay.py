from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

from .settings import SERVER

log = logging.getLogger("automate.relay")


def run(
    relay_url: str,
    token: str | None = None,
    *,
    local_url: str | None = None,
) -> int:
    try:
        import httpx  # type: ignore
        import websockets  # type: ignore
    except ImportError:
        sys.stderr.write(
            "The relay client needs websockets + httpx. Install with:\n"
            "  pip install 'automate-hub[relay]'\n"
        )
        return 1

    destination = (local_url or SERVER.base_url).rstrip("/")
    relay_token = token or os.environ.get("AUTOMATE_RELAY_TOKEN", "")

    async def connect() -> None:
        headers = (
            {"Authorization": f"Bearer {relay_token}"}
            if relay_token
            else {}
        )

        async with httpx.AsyncClient(base_url=destination, timeout=60) as client:
            print(f"  relay  →  dialing {relay_url}", flush=True)

            async with websockets.connect(
                relay_url,
                additional_headers=headers,
            ) as socket:
                await socket.send(
                    json.dumps({"hello": "automate-hub", "version": "1"})
                )
                print(
                    f"  relay  →  connected; forwarding to {destination}",
                    flush=True,
                )

                async for message in socket:
                    try:
                        payload = json.loads(message)
                    except Exception:
                        continue
                    asyncio.create_task(_handle_frame(socket, client, payload))

    try:
        asyncio.run(connect())
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        log.error("relay disconnected: %s", error)
        return 1

    return 0


async def _handle_frame(ws, http, frame: dict[str, Any]) -> None:
    request_id = frame.get("id")
    request_kind = frame.get("kind")

    if request_kind != "http":
        await ws.send(
            json.dumps(
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"unknown kind: {request_kind}",
                }
            )
        )
        return

    method = frame.get("method", "GET").upper()
    path = frame.get("path", "/")
    headers = frame.get("headers") or {}
    body = frame.get("body")

    try:
        response = await http.request(
            method,
            path,
            headers=headers,
            content=body,
        )
        await ws.send(
            json.dumps(
                {
                    "id": request_id,
                    "ok": True,
                    "status": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                }
            )
        )
    except Exception as error:
        await ws.send(
            json.dumps(
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        )