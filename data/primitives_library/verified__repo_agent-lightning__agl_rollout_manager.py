from __future__ import annotations

import json
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from httpx_retries import Retry, RetryTransport
from pydantic import BaseModel, Field

from agentlightning.client import AgentLightningSyncClient
from agentlightning.schemas import (
    TERMINAL_STATES,
    Event,
    EventCreate,
    Model,
    Rollout,
    RolloutCreate,
    RolloutState,
)

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    from agentlightning.hooks import RolloutHooks


class Triplet(BaseModel):
    """Single prompt-response-reward turn."""

    prompt: Any
    response: Any
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_urls: list[str] | None = None


class EnqueuedRollout(BaseModel):
    """Enqueued rollout request metadata."""

    data_id: str
    rollout_id: str
    step: int
    sample_idx_in_step: int
    enqueue_time: float
    input: Any = None
    running_at: float | None = None
    finished_at: float | None = None


class CompletedRollout(BaseModel):
    """Completed rollout result."""

    rollout_id: str
    data_id: str
    step: int
    sample_idx_in_step: int
    enqueue_time: float
    input: Any = None
    running_at: float | None = None
    finished_at: float | None = None
    final_reward: float | None = None
    triplets: list[Triplet] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    triplet_events: list[dict[str, Any]] = Field(default_factory=list)
    rollout_state: RolloutState | None = None
    error_message: str | None = None


@dataclass
class _TraceEvent:
    rollout_id: str
    attempt_id: str
    event_type: str
    data: dict[str, Any]


class _TraceEventHelper:
    """Queues hook events before HTTP flush."""

    def __init__(self) -> None:
        self._queued: list[_TraceEvent] = []

    def add_event(
        self,
        rollout_id: str,
        attempt_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        self._queued.append(
            _TraceEvent(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                event_type=event_type,
                data=data,
            )
        )

    def flush(self, manager: "AglRolloutManagerBase") -> None:
        for event in self._queued:
            manager._post_event(
                event.rollout_id,
                event.attempt_id,
                EventCreate(event_type=event.event_type, data=event.data),
            )
        self._queued.clear()


def _as_reward_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _to_native(obj: Any) -> Any:
    """Convert numpy/torch values for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_native(obj.item())
    if isinstance(obj, Mapping):
        return {_to_native(key): _to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(item) for item in obj]
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    return obj


def _extract_image_urls_from_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    urls: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                urls.append(image_url["url"])
    return urls


def _aligned_image_urls(
    raw_events: list[Event],
    n_triplets: int,
) -> list[list[str] | None] | None:
    requests: list[tuple[dict[str, Any], list[int], list[int], list[str]]] = []
    for event in raw_events:
        if event.event_type != "model_request":
            continue
        data = event.data
        request = data.get("request")
        messages = request.get("messages") if isinstance(request, dict) else None
        image_urls = _extract_image_urls_from_messages(messages)

        prompt_token_ids: Any = []
        response_token_ids: Any = []
        response = data.get("response")
        if isinstance(response, dict):
            prompt_token_ids = response.get("prompt_token_ids", [])
            choices = response.get("choices", [])
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    if not prompt_token_ids:
                        prompt_token_ids = first.get("prompt_token_ids", [])
                    response_token_ids = first.get("token_ids", [])
        elif isinstance(response, list):
            for chunk in response:
                if not isinstance(chunk, dict):
                    continue
                if not prompt_token_ids and chunk.get("prompt_token_ids"):
                    prompt_token_ids = chunk["prompt_token_ids"]
                choices = chunk.get("choices", [])
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    token_ids = choices[0].get("token_ids")
                    if token_ids:
                        if not isinstance(response_token_ids, list):
                            response_token_ids = []
                        response_token_ids.extend(token_ids)

        if not isinstance(prompt_token_ids, list):
            prompt_token_ids = []
        if not isinstance(response_token_ids, list):
            response_token_ids = []
        requests.append((data, prompt_token_ids, response_token_ids, image_urls))

    if not any(urls for _, _, _, urls in requests):
        return None

    last_for_prompt: dict[tuple[int, ...], int] = {}
    retained: set[int] = set()
    for index, (_, prompt_ids, _, _) in enumerate(requests):
        if not prompt_ids or any(type(token) is not int for token in prompt_ids):
            retained.add(index)
        else:
            last_for_prompt[tuple(prompt_ids)] = index
    retained.update(last_for_prompt.values())

    output: list[list[str] | None] = []
    for index, (data, _, response_ids, urls) in enumerate(requests):
        if index not in retained:
            continue
        status = data.get("http_status")
        if data.get("status") == "error" or (
            isinstance(status, int) and status >= 400
        ):
            continue
        if not response_ids:
            continue
        output.append(urls or None)

    if len(output) != n_triplets:
        return None
    return output


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return cast(dict[str, Any], value.dict())
    return {}


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _event_data(event: Any) -> dict[str, Any]:
    data = _get(event, "data", {})
    return dict(data) if isinstance(data, Mapping) else {}


def _event_type(event: Any) -> str | None:
    value = _get(event, "event_type")
    return value if isinstance(value, str) else None


class AglRolloutManagerBase:
    """Common client-side bookkeeping for Agent Lightning rollouts."""

    def __init__(
        self,
        client: AgentLightningSyncClient | None = None,
        model: Model | str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        rollout_hooks: "RolloutHooks | None" = None,
        retry_count: int = 3,
        retry_backoff_factor: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if client is None:
            transport = kwargs.pop("transport", None)
            if transport is None:
                try:
                    transport = RetryTransport(
                        retry=Retry(
                            total=retry_count,
                            backoff_factor=retry_backoff_factor,
                        )
                    )
                except Exception:
                    transport = None
            client_kwargs: dict[str, Any] = dict(kwargs)
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            if transport is not None:
                client_kwargs["transport"] = transport
            try:
                client = AgentLightningSyncClient(**client_kwargs)
            except TypeError:
                client = AgentLightningSyncClient()
        self.client = client
        self.model = model
        self.rollout_hooks = rollout_hooks
        self._enqueued_rollouts: dict[str, EnqueuedRollout] = {}
        self._completed_rollouts: list[CompletedRollout] = []
        self._rollout_attempt_ids: dict[str, str] = {}
        self._pending_rollout_ids: set[str] = set()

    @property
    def enqueued_rollouts(self) -> dict[str, EnqueuedRollout]:
        return self._enqueued_rollouts

    @property
    def completed_rollouts(self) -> list[CompletedRollout]:
        return self._completed_rollouts

    def _call_client(self, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            target: Any = self.client
            found = True
            for part in name.split("."):
                if not hasattr(target, part):
                    found = False
                    break
                target = getattr(target, part)
            if not found or not callable(target):
                continue
            try:
                return target(*args, **kwargs)
            except TypeError:
                try:
                    return target(*args)
                except TypeError:
                    continue
        raise AttributeError(f"Client exposes none of: {', '.join(names)}")

    def _post_event(
        self,
        rollout_id: str,
        attempt_id: str,
        event: EventCreate,
    ) -> Any:
        return self._call_client(
            (
                "create_event",
                "post_event",
                "events.create",
            ),
            rollout_id,
            attempt_id,
            event,
        )

    def _create_rollout(self, request: RolloutCreate) -> Any:
        return self._call_client(
            (
                "create_rollout",
                "post_rollout",
                "rollouts.create",
            ),
            request,
        )

    def _get_rollout(self, rollout_id: str) -> Any:
        return self._call_client(
            (
                "get_rollout",
                "retrieve_rollout",
                "rollouts.get",
                "rollouts.retrieve",
            ),
            rollout_id,
        )

    def _get_events(self, rollout_id: str, attempt_id: str | None = None) -> list[Any]:
        for args in (
            (rollout_id, attempt_id),
            (rollout_id,),
        ):
            try:
                result = self._call_client(
                    (
                        "list_events",
                        "get_events",
                        "events.list",
                    ),
                    *args,
                )
                if isinstance(result, Mapping):
                    result = result.get("items", result.get("data", []))
                return list(result or [])
            except (AttributeError, TypeError):
                pass
        return []

    def enqueue_rollout(
        self,
        data_id: str,
        input: Any = None,
        step: int = 0,
        sample_idx_in_step: int = 0,
        **kwargs: Any,
    ) -> EnqueuedRollout:
        payload: dict[str, Any] = {
            "data_id": data_id,
            "input": _to_native(input),
        }
        if self.model is not None:
            payload["model"] = self.model
        payload.update({key: _to_native(value) for key, value in kwargs.items()})
        try:
            request = RolloutCreate(**payload)
        except Exception:
            payload.pop("data_id", None)
            request = RolloutCreate(**payload)
        rollout = self._create_rollout(request)
        rollout_id = str(_get(rollout, "id", _get(rollout, "rollout_id", uuid.uuid4())))
        attempt_id = _get(rollout, "attempt_id")
        if attempt_id is not None:
            self._rollout_attempt_ids[rollout_id] = str(attempt_id)
        queued = EnqueuedRollout(
            data_id=str(data_id),
            rollout_id=rollout_id,
            step=step,
            sample_idx_in_step=sample_idx_in_step,
            enqueue_time=time.time(),
            input=input,
        )
        self._enqueued_rollouts[rollout_id] = queued
        self._pending_rollout_ids.add(rollout_id)
        return queued

    def enqueue_rollouts(
        self,
        data_ids: Iterable[str],
        inputs: Iterable[Any] | None = None,
        step: int = 0,
        **kwargs: Any,
    ) -> list[EnqueuedRollout]:
        ids = list(data_ids)
        values = list(inputs) if inputs is not None else [None] * len(ids)
        if len(values) != len(ids):
            raise ValueError("data_ids and inputs must have equal lengths")
        return [
            self.enqueue_rollout(
                data_id,
                value,
                step=step,
                sample_idx_in_step=index,
                **kwargs,
            )
            for index, (data_id, value) in enumerate(zip(ids, values, strict=True))
        ]

    def _build_completed_rollout(
        self,
        rollout: Rollout | Any,
        enqueued: EnqueuedRollout,
        events: list[Event] | list[Any] | None = None,
    ) -> CompletedRollout:
        raw_events = list(events if events is not None else [])
        event_dicts = [_dump(event) for event in raw_events]
        triplets: list[Triplet] = []
        triplet_events: list[dict[str, Any]] = []
        final_reward = _as_reward_value(_get(rollout, "reward"))
        metadata = _get(rollout, "metadata", {})
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

        for event, event_dict in zip(raw_events, event_dicts, strict=True):
            event_type = _event_type(event)
            data = _event_data(event)
            if event_type == "model_request":
                status = data.get("http_status")
                if data.get("status") == "error" or (
                    isinstance(status, int) and status >= 400
                ):
                    continue
                response = data.get("response")
                request = data.get("request")
                prompt: Any = None
                completion: Any = None
                if isinstance(request, Mapping):
                    prompt = request.get("messages", request.get("prompt"))
                if isinstance(response, Mapping):
                    choices = response.get("choices", [])
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        if isinstance(choice, Mapping):
                            completion = choice.get(
                                "message",
                                choice.get("text", choice.get("content")),
                            )
                            if completion is None:
                                completion = choice.get("token_ids")
                    if prompt is None:
                        prompt = response.get("prompt_token_ids")
                elif isinstance(response, list):
                    token_ids: list[Any] = []
                    for chunk in response:
                        if not isinstance(chunk, Mapping):
                            continue
                        if prompt is None and chunk.get("prompt_token_ids") is not None:
                            prompt = chunk.get("prompt_token_ids")
                        choices = chunk.get("choices", [])
                        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                            choice = choices[0]
                            if choice.get("token_ids") is not None:
                                token_ids.extend(choice["token_ids"])
                            elif completion is None:
                                completion = choice.get("text", choice.get("content"))
                    if token_ids:
                        completion = token_ids
                if completion is None:
                    continue
                triplets.append(
                    Triplet(
                        prompt=prompt,
                        response=completion,
                        metadata={
                            key: _to_native(value)
                            for key, value in data.items()
                            if key not in {"request", "response"}
                        },
                    )
                )
                triplet_events.append(event_dict)
            elif event_type in {"reward", "rollout_reward", "final_reward"}:
                reward = _as_reward_value(data.get("reward", data.get("value")))
                if reward is not None:
                    final_reward = reward

        image_urls = _aligned_image_urls(
            [cast(Event, event) for event in raw_events],
            len(triplets),
        )
        if image_urls is not None:
            for triplet, urls in zip(triplets, image_urls, strict=True):
                triplet.image_urls = urls

        state = _get(rollout, "state")
        try:
            rollout_state = state if isinstance(state, RolloutState) else RolloutState(state)
        except Exception:
            rollout_state = None

        result = _get(rollout, "result")
        if isinstance(result, Mapping):
            reward = _as_reward_value(result.get("reward", result.get("final_reward")))
            if reward is not None:
                final_reward = reward
            result_metadata = result.get("metadata")
            if isinstance(result_metadata, Mapping):
                metadata.update(_to_native(result_metadata))

        error_message = _get(rollout, "error_message", _get(rollout, "error"))
        if error_message is not None and not isinstance(error_message, str):
            error_message = str(error_message)

        return CompletedRollout(
            rollout_id=enqueued.rollout_id,
            data_id=enqueued.data_id,
            step=enqueued.step,
            sample_idx_in_step=enqueued.sample_idx_in_step,
            enqueue_time=enqueued.enqueue_time,
            input=enqueued.input,
            running_at=_get(rollout, "running_at", enqueued.running_at),
            finished_at=_get(rollout, "finished_at", enqueued.finished_at),
            final_reward=final_reward,
            triplets=triplets or None,
            metadata=_to_native(metadata),
            events=event_dicts,
            triplet_events=triplet_events,
            rollout_state=rollout_state,
            error_message=error_message,
        )

    def poll_completed_rollouts(self) -> list[CompletedRollout]:
        completed: list[CompletedRollout] = []
        for rollout_id in list(self._pending_rollout_ids):
            try:
                rollout = self._get_rollout(rollout_id)
            except Exception:
                continue
            state = _get(rollout, "state")
            try:
                terminal = state in TERMINAL_STATES or RolloutState(state) in TERMINAL_STATES
            except Exception:
                terminal = False
            if not terminal:
                queued = self._enqueued_rollouts.get(rollout_id)
                if queued is not None:
                    queued.running_at = _get(rollout, "running_at", queued.running_at)
                continue
            queued = self._enqueued_rollouts.get(rollout_id)
            if queued is None:
                continue
            attempt_id = _get(rollout, "attempt_id", self._rollout_attempt_ids.get(rollout_id))
            events = self._get_events(rollout_id, str(attempt_id) if attempt_id else None)
            item = self._build_completed_rollout(rollout, queued, events)
            completed.append(item)
            self._completed_rollouts.append(item)
            self._pending_rollout_ids.discard(rollout_id)
        return completed

    def get_completed_rollouts(self) -> list[CompletedRollout]:
        return self.poll_completed_rollouts()

    def wait_for_rollouts(
        self,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> list[CompletedRollout]:
        started = time.monotonic()
        result: list[CompletedRollout] = []
        while self._pending_rollout_ids:
            result.extend(self.poll_completed_rollouts())
            if not self._pending_rollout_ids:
                break
            if timeout is not None and time.monotonic() - started >= timeout:
                break
            time.sleep(poll_interval)
        return result


class AglRolloutManager(AglRolloutManagerBase):
    """Default Agent Lightning rollout manager used by VERL."""


__all__ = [
    "AglRolloutManager",
    "AglRolloutManagerBase",
    "CompletedRollout",
    "EnqueuedRollout",
    "Triplet",
]