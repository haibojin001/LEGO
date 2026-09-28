from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...agent import RunEvent

router = APIRouter(tags=["sessions"], prefix="/sessions")


@router.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    state = websocket.app.state.app_state

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"kind": "error", "payload": {"message": "invalid JSON"}}
                )
                continue

            prompt = msg.get("prompt", "")
            model = msg.get("model")
            allowed = msg.get("allowed_tools")

            if not prompt.strip():
                await websocket.send_json(
                    {"kind": "error", "payload": {"message": "prompt is required"}}
                )
                continue

            loop = asyncio.get_running_loop()

            async def emit(ev: RunEvent):
                await websocket.send_json({"kind": ev.kind, "payload": ev.payload})

            def emit_sync(ev: RunEvent):
                asyncio.run_coroutine_threadsafe(emit(ev), loop)

            await loop.run_in_executor(
                None,
                lambda: state.agent.run(
                    prompt,
                    source="ws",
                    model=model,
                    allowed_tools=allowed,
                    on_event=emit_sync,
                ),
            )
            await websocket.send_json({"kind": "done", "payload": {}})
    except WebSocketDisconnect:
        return