from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from agentlightning.schemas import Model
from agentlightning.server.routes.events import record_event
from agentlightning.server.store import _models

log = structlog.get_logger()

_UPSTREAM_MAX_ATTEMPTS = 6
_RETRY_STATUS_CODES = {408, 409, 429}
_RETRY_BACKOFF_BASE_SECONDS = 0.5
_RETRY_BACKOFF_CAP_SECONDS = 8.0


class NoServersError(Exception):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"No servers available for model '{model}'")


class ProxyRouter:
    """Routes requests to model servers and applies proxy configuration."""

    def __init__(self, default_proxy: Mapping[str, Any]) -> None:
        self._model_name = str(default_proxy["model_name"])
        self._train_temperature = float(default_proxy["train"]["temperature"])
        self._val_temperature = float(default_proxy["val"]["temperature"])
        self._include_log_probs = bool(default_proxy.get("include_log_probs", True))

    @property
    def model_name(self) -> str:
        return self._model_name

    def select_server(self, model: str, rollout_id: str) -> Model:
        registered = _models.get(model, {})
        if not registered:
            raise NoServersError(model)

        ordered_servers = [registered[key] for key in sorted(registered)]
        hash_bytes = hashlib.sha256(rollout_id.encode("utf-8")).digest()
        selected_index = int.from_bytes(hash_bytes[:8], byteorder="big") % len(ordered_servers)
        return ordered_servers[selected_index]

    def prepare_body(self, body: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode == "train":
            result = dict(body)
            result["model"] = self._model_name
            result["temperature"] = self._train_temperature
            result["return_token_ids"] = True
            if self._include_log_probs:
                result["logprobs"] = True
            return result

        if mode == "val":
            result = dict(body)
            result["model"] = self._model_name
            result["temperature"] = self._val_temperature
            result["return_token_ids"] = True
            return result

        raise ValueError(f"Unsupported proxy mode: {mode}")


@dataclass
class ProxyPauseState:
    paused: bool = False
    retry_after_seconds: int = 5
    reason: str | None = None
    inflight: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def forward_request(
    *,
    client: httpx.AsyncClient,
    server: Model,
    body: dict[str, Any],
    upstream_path: str = "chat/completions",
    rollout_id: str,
    attempt_id: str,
    pause_state: ProxyPauseState | None = None,
) -> Response:
    if pause_state is not None:
        async with pause_state.lock:
            if pause_state.paused:
                return Response(
                    content=json.dumps(
                        {
                            "error": "gateway paused",
                            "reason": pause_state.reason,
                        }
                    ),
                    status_code=429,
                    headers={
                        "Retry-After": str(pause_state.retry_after_seconds),
                        "X-Agl-Paused": "true",
                    },
                    media_type="application/json",
                )
            pause_state.inflight += 1

    try:
        if body.get("stream", False):
            raise HTTPException(status_code=400, detail="Streaming responses are not supported")

        target_url = f"{server.endpoint.rstrip('/')}/{upstream_path}"
        log.debug(
            "Proxying request",
            rollout_id=rollout_id,
            model=server.model,
            path=upstream_path,
        )

        start_time = time.perf_counter()
        upstream_response = await _send_upstream_with_retries(
            client=client,
            url=target_url,
            body=body,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        content_type = upstream_response.headers.get("content-type", "")
        parsed_body = upstream_response.json() if content_type.startswith("application/json") else {}

        _capture_event(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            request_body=body,
            response_body=parsed_body,
            server=server,
            latency_ms=duration_ms,
            http_status=upstream_response.status_code,
            status=_status_from_http_status(upstream_response.status_code),
            retry_count=int(upstream_response.extensions.get("agl_retry_count", 0)),
        )

        return JSONResponse(
            content=parsed_body,
            status_code=upstream_response.status_code,
        )
    finally:
        if pause_state is not None:
            await _dec_inflight(pause_state)


async def _send_upstream_with_retries(
    *,
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
) -> httpx.Response:
    for attempt_index in range(_UPSTREAM_MAX_ATTEMPTS):
        try:
            response = await client.post(
                url,
                json=body,
                headers={"content-type": "application/json"},
            )
        except httpx.TimeoutException as error:
            if attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
                raise HTTPException(
                    status_code=504,
                    detail="Upstream model server timed out",
                ) from error
            await _sleep_before_retry(
                url=url,
                attempt_index=attempt_index,
                reason="timeout",
            )
            continue
        except httpx.TransportError as error:
            if attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream model server request failed",
                ) from error
            await _sleep_before_retry(
                url=url,
                attempt_index=attempt_index,
                reason="transport error",
            )
            continue

        if not _is_retryable_status(response.status_code) or attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
            response.extensions["agl_retry_count"] = attempt_index
            return response

        await response.aclose()
        await _sleep_before_retry(
            url=url,
            attempt_index=attempt_index,
            reason=f"status {response.status_code}",
        )

    raise HTTPException(status_code=502, detail="Upstream model server request failed")


async def _sleep_before_retry(*, url: str, attempt_index: int, reason: str) -> None:
    wait_time = _retry_delay_seconds(attempt_index)
    log.warning(
        "Retrying upstream request",
        url=url,
        attempt=attempt_index + 1,
        max_attempts=_UPSTREAM_MAX_ATTEMPTS,
        delay_seconds=round(wait_time, 3),
        reason=reason,
    )
    await asyncio.sleep(wait_time)


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRY_STATUS_CODES or status_code >= 500


def _retry_delay_seconds(attempt_index: int) -> float:
    backoff = min(
        _RETRY_BACKOFF_BASE_SECONDS * (2**attempt_index),
        _RETRY_BACKOFF_CAP_SECONDS,
    )
    return backoff * random.uniform(0.75, 1.25)


async def _dec_inflight(pause_state: ProxyPauseState) -> None:
    async with pause_state.lock:
        pause_state.inflight = max(0, pause_state.inflight - 1)


def _capture_event(
    *,
    rollout_id: str,
    attempt_id: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    server: Model,
    latency_ms: float,
    http_status: int,
    status: str,
    retry_count: int,
) -> None:
    record_event(
        rollout_id,
        attempt_id,
        "model_request",
        {
            "model": server.model,
            "model_version": server.version,
            "request": request_body,
            "response": response_body,
            "server": {
                "model": server.model,
                "endpoint": server.endpoint,
                "version": server.version,
            },
            "latency_ms": latency_ms,
            "http_status": http_status,
            "status": status,
            "retry_count": retry_count,
            "usage": _extract_usage(response_body),
            "finish_reason": _extract_finish_reason(response_body),
        },
    )


def _status_from_http_status(http_status: int) -> str:
    if http_status < 400:
        return "ok"
    return "error"


def _extract_usage(response_body: dict[str, Any]) -> dict[str, Any] | None:
    candidate = response_body.get("usage")
    if isinstance(candidate, dict):
        return candidate
    return None


def _extract_finish_reason(response_body: dict[str, Any]) -> str | None:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if not isinstance(first, dict):
        return None

    finish_reason = first.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason
    return None