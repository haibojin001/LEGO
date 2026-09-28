from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import HTTPException
from omegaconf import DictConfig, OmegaConf

from agentlightning.server.proxy import ProxyPauseState, ProxyRouter
from agentlightning.server.routes import events, models, proxy, rollouts

log = structlog.get_logger()


def _server_config(config: Mapping[str, Any] | DictConfig | None) -> dict[str, Any]:
    if config is None:
        raise ValueError("server config is required")

    if OmegaConf.is_config(config):
        container = OmegaConf.to_container(config, resolve=True)
        return dict(cast(Any, container))

    return dict(config)


def _build_auth_dependency(key: str):
    async def verify_key(request: Request) -> None:
        if not key:
            return

        authorization = request.headers.get("authorization", "")
        if authorization.startswith("Bearer ") and authorization[7:] == key:
            return

        if request.headers.get("x-api-key", "") == key:
            return

        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_key


def create_app(config: Mapping[str, Any] | DictConfig | None = None) -> FastAPI:
    server_config = _server_config(config)
    key = str(server_config["key"] or "")

    if not key:
        log.warning("AGL_KEY not set — authentication disabled. Do not use in production.")

    verify_key = _build_auth_dependency(key)
    default_proxy = server_config["default_proxy"]
    log.info("Proxy config loaded", model_name=default_proxy["model_name"])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.proxy_pause_state = ProxyPauseState()
        app.state.proxy_router = ProxyRouter(default_proxy)

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=300.0)) as client:
            app.state.http_client = client
            yield

    app = FastAPI(title="Agent Lightning", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    dependencies = [Depends(verify_key)]
    app.include_router(rollouts.router, prefix="/api", dependencies=dependencies)
    app.include_router(events.router, prefix="/api", dependencies=dependencies)
    app.include_router(models.router, prefix="/api", dependencies=dependencies)
    app.include_router(proxy.router, dependencies=dependencies)
    app.include_router(proxy.management_router, dependencies=dependencies)

    return app