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
        bearer_valid = (
            authorization.startswith("Bearer ") and authorization[7:] == key
        )
        api_key_valid = request.headers.get("x-api-key", "") == key

        if bearer_valid or api_key_valid:
            return

        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_key


def create_app(config: Mapping[str, Any] | DictConfig | None = None) -> FastAPI:
    settings = _server_config(config)
    api_key = str(settings["key"] or "")

    if not api_key:
        log.warning(
            "AGL_KEY not set — authentication disabled. Do not use in production."
        )

    auth_dependency = _build_auth_dependency(api_key)
    proxy_settings = settings["default_proxy"]
    log.info("Proxy config loaded", model_name=proxy_settings["model_name"])

    @asynccontextmanager
    async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.proxy_pause_state = ProxyPauseState()
        application.state.proxy_router = ProxyRouter(proxy_settings)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=300.0)
        ) as client:
            application.state.http_client = client
            yield

    application = FastAPI(
        title="Agent Lightning",
        version="1.0.0",
        lifespan=application_lifespan,
    )

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    protected = [Depends(auth_dependency)]
    application.include_router(rollouts.router, prefix="/api", dependencies=protected)
    application.include_router(events.router, prefix="/api", dependencies=protected)
    application.include_router(models.router, prefix="/api", dependencies=protected)
    application.include_router(proxy.router, dependencies=protected)
    application.include_router(proxy.management_router, dependencies=protected)

    return application