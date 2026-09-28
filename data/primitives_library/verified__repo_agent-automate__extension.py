from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...extension_bus import bus
from ...version import __version__

router = APIRouter(tags=["extension"], prefix="/extension")


@router.get("/status")
def status():
    return bus.status


@router.websocket("/ws")
async def extension_ws(ws: WebSocket):
    await ws.accept()

    if bus.connected:
        await ws.send_json({"hello": "automate", "error": "already connected"})
        await ws.close(code=1008)
        return

    loop = asyncio.get_running_loop()
    bus.attach(
        loop,
        client_info={
            "user_agent": ws.headers.get("user-agent", ""),
            "origin": ws.headers.get("origin", ""),
        },
    )

    await ws.send_json({"hello": "automate", "version": __version__})

    async def push_outgoing():
        while bus.connected:
            try:
                message = await bus.next_outgoing()
            except RuntimeError:
                break
            await ws.send_text(message)

    sender = asyncio.create_task(push_outgoing())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if message.get("hello"):
                continue

            bus.deliver(message)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        bus.detach()