from __future__ import annotations

import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import channels as C
from ..version import __version__
from .api import (
    agent,
    audio,
    auth,
    bots,
    channels as channels_api,
    execute,
    extension,
    files,
    integrations,
    memory,
    models,
    notes,
    oauth,
    push,
    reminders,
    sessions,
    system,
    tools as tools_api,
)
from .state import AppState, build_state


def create_app() -> FastAPI:
    application_state = build_state()

    try:
        from .mcp_bridge import build_http_app

        mcp_application, mcp_session_manager = build_http_app(application_state)
    except SystemExit:
        mcp_application = None
        mcp_session_manager = None

    @contextlib.asynccontextmanager
    async def app_lifespan(_: FastAPI):
        if mcp_session_manager is None:
            yield
        else:
            async with mcp_session_manager.run():
                yield

    application = FastAPI(
        title="autoMate",
        version=__version__,
        description=(
            "A smart NAS for AI — notes, files, reminders, memory, tools. "
            "Plug in any LLM via MCP / HTTP / bridge."
        ),
        lifespan=app_lifespan,
    )
    application.state.app_state = application_state

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    routers = (
        (system.router, "/api"),
        (models.router, "/api"),
        (tools_api.router, "/api"),
        (integrations.router, "/api"),
        (agent.router, "/api"),
        (execute.router, "/api"),
        (sessions.router, "/api"),
        (extension.router, "/api"),
        (notes.router, "/api"),
        (files.router, "/api"),
        (reminders.router, "/api"),
        (memory.router, "/api"),
        (push.router, "/api"),
        (bots.router, "/api"),
        (auth.router, "/api"),
        (audio.router, "/api"),
        (channels_api.router, "/api"),
    )
    for router, prefix in routers:
        application.include_router(router, prefix=prefix)

    application.include_router(oauth.router)

    if mcp_application is not None:
        application.mount(
            "/mcp",
            _BearerProtect(mcp_application, application_state.db),
        )

    frontend_path = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_path.exists():
        application.mount(
            "/",
            StaticFiles(directory=str(frontend_path), html=True),
            name="frontend",
        )

    return application


class _BearerProtect:
    def __init__(self, inner, db):
        self.inner = inner
        self.db = db

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return

        if scope.get("path", "") == "":
            await _redirect(send, "/mcp/")
            return

        headers = {
            name.decode().lower(): value.decode()
            for name, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization", "")

        if not authorization.lower().startswith("bearer "):
            await _reject(send, 401, "missing Bearer token")
            return

        token = authorization.split(" ", 1)[1].strip()
        if not C.verify_token(self.db, token):
            await _reject(send, 401, "invalid token")
            return

        await self.inner(scope, receive, send)


async def _redirect(send, location: str) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 307,
            "headers": [
                (b"location", location.encode()),
                (b"content-length", b"0"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b""})


async def _reject(send, status: int, detail: str) -> None:
    body = b'{"error":"' + detail.encode() + b'"}'
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def get_state(app: FastAPI) -> AppState:
    return app.state.app_state