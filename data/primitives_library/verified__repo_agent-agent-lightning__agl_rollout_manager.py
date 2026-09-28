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
    """A single prompt, response and optional reward."""

    prompt: Any
    response: Any
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_urls: list[str] | None = None


class EnqueuedRollout(BaseModel):
    """Information retained for a submitted rollout."""

    data_id: str
    rollout_id: str
    step: int
    sample_idx_in_step: int
    enqueue_time: float
    input: Any = None
    running_at: float | None = None
    finished_at: float | None = None


class CompletedRollout(BaseModel):
    """A rollout and the trajectory extracted from it."""

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
    """Collect hook events until it is safe to send them to the server."""

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
        for item in self._queued:
            manager._post_event(
                item.rollout_id,
                item.attempt_id,
                EventCreate(event_type=item.event_type, data=item.data),
            )
        self._queued.clear()


def _as_reward_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _to_native(obj: Any) -> Any:
    """Make numpy and torch values suitable for JSON transport."""
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_native(obj.item())
    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return _to_native(obj.item())
        return _to_native(obj.tolist())
    if isinstance(obj, Mapping):
        return {_to_native(key): _to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(value) for value in obj]
    return obj


def _extract_image_urls_from_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    output: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (TypeError, ValueError):
                continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "image_url":
                continue
            image = part.get("image_url")
            if isinstance(image, Mapping) and isinstance(image.get("url"), str):
                output.append(image["url"])
    return output


def _event_data(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        value = event.get("data", {})
    else:
        value = getattr(event, "data", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _event_type(event: Any) -> str | None:
    if isinstance(event, Mapping):
        return cast(str | None, event.get("event_type"))
    return cast(str | None, getattr(event, "event_type", None))


def _aligned_image_urls(raw_events: list[Event], n_triplets: int) -> list[list[str] | None] | None:
    requests: list[tuple[dict[str, Any], list[int], list[int], list[str]]] = []
    for event in raw_events:
        if _event_type(event) != "model_request":
            continue
        data = _event_data(event)
        request = data.get("request")
        messages = request.get("messages") if isinstance(request, Mapping) else None
        images = _extract_image_urls_from_messages(messages)

        prompt_ids: Any = []
        response_ids: Any = []
        response = data.get("response")
        if isinstance(response, Mapping):
            prompt_ids = response.get("prompt_token_ids", [])
            choices = response.get("choices", [])
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    if not prompt_ids:
                        prompt_ids = choice.get("prompt_token_ids", [])
                    response_ids = choice.get("token_ids", [])
        elif isinstance(response, list):
            for chunk in response:
                if not isinstance(chunk, Mapping):
                    continue
                if not prompt_ids and chunk.get("prompt_token_ids"):
                    prompt_ids = chunk["prompt_token_ids"]
                choices = chunk.get("choices", [])
                if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                    token_ids = choices[0].get("token_ids")
                    if isinstance(token_ids, list):
                        if not isinstance(response_ids, list):
                            response_ids = []
                        response_ids.extend(token_ids)

        if not isinstance(prompt_ids, list):
            prompt_ids = []
        if not isinstance(response_ids, list):
            response_ids = []
        requests.append((data, prompt_ids, response_ids, images))

    if not any(images for _, _, _, images in requests):
        return None

    last_by_prompt: dict[tuple[int, ...], int] = {}
    keep: set[int] = set()
    for index, (_, prompt_ids, _, _) in enumerate(requests):
        if not prompt_ids or any(type(token) is not int for token in prompt_ids):
            keep.add(index)
        else:
            last_by_prompt[tuple(prompt_ids)] = index
    keep.update(last_by_prompt.values())

    output: list[list[str] | None] = []
    for index, (data, _, response_ids, images) in enumerate(requests):
        if index not in keep:
            continue
        status = data.get("http_status")
        if data.get("status") == "error" or (isinstance(status, int) and status >= 400):
            continue
        if not response_ids:
            continue
        output.append(images or None)

    if len(output) != n_triplets:
        return None
    return output


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _to_native(dict(value))
    if hasattr(value, "model_dump"):
        return _to_native(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _to_native(value.dict())
    return {}


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class AglRolloutManagerBase:
    """Base implementation shared by VERL rollout managers.

    The manager deliberately keeps only a small local queue.  The rollout service is
    authoritative for state and trajectory information, which makes the object safe to
    recreate between trainer invocations.
    """

    def __init__(
        self,
        client: AgentLightningSyncClient | None = None,
        *,
        base_url: str | None = None,
        agentlightning_url: str | None = None,
        model: Model | str | None = None,
        model_id: str | None = None,
        project_name: str | None = None,
        experiment_name: str | None = None,
        rollout_hooks: "RolloutHooks | None" = None,
        hooks: "RolloutHooks | None" = None,
        output_dir: str | Path | None = None,
        retry: Retry | None = None,
        **kwargs: Any,
    ) -> None:
        if client is None:
            url = base_url or agentlightning_url or kwargs.pop("url", None)
            transport = None
            if retry is not None:
                transport = RetryTransport(retry=retry)
            elif kwargs.pop("enable_retries", True):
                try:
                    transport = RetryTransport(retry=Retry(total=3))
                except Exception:
                    transport = None
            client_kwargs = dict(kwargs)
            if url is not None:
                client_kwargs.setdefault("base_url", url)
            if transport is not None:
                client_kwargs.setdefault("transport", transport)
            client = AgentLightningSyncClient(**client_kwargs)
        self.client = client
        self.model = model
        self.model_id = model_id or (
            _read_field(model, "id") if model is not None and not isinstance(model, str) else model
        )
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.rollout_hooks = rollout_hooks or hooks
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._enqueued_rollouts: dict[str, EnqueuedRollout] = {}
        self._completed_rollout_ids: set[str] = set()
        self._trace_events = _TraceEventHelper()

    @property
    def enqueued_rollouts(self) -> dict[str, EnqueuedRollout]:
        return self._enqueued_rollouts

    @property
    def pending_rollouts(self) -> list[EnqueuedRollout]:
        return list(self._enqueued_rollouts.values())

    def _call_client(self, names: Iterable[str], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            method = getattr(self.client, name, None)
            if method is None:
                continue
            try:
                return method(*args, **kwargs)
            except TypeError:
                if args:
                    try:
                        return method(**kwargs)
                    except TypeError:
                        pass
                raise
        raise AttributeError(f"Client has none of: {', '.join(names)}")

    def _create_rollout(self, payload: Any) -> Any:
        return self._call_client(("create_rollout", "post_rollout"), payload)

    def _get_rollout(self, rollout_id: str) -> Any:
        return self._call_client(("get_rollout", "retrieve_rollout"), rollout_id)

    def _get_events(self, rollout_id: str) -> list[Any]:
        result = self._call_client(
            ("list_events", "get_events", "retrieve_events"),
            rollout_id,
        )
        if isinstance(result, Mapping):
            for key in ("items", "data", "events"):
                if isinstance(result.get(key), list):
                    return list(result[key])
        if hasattr(result, "items") and not isinstance(result, list):
            items = getattr(result, "items")
            if isinstance(items, list):
                return items
        return list(result or [])

    def _post_event(self, rollout_id: str, attempt_id: str, event: EventCreate) -> Any:
        methods = ("create_event", "post_event", "append_event")
        for name in methods:
            method = getattr(self.client, name, None)
            if method is None:
                continue
            for args in (
                (rollout_id, attempt_id, event),
                (rollout_id, event),
                (),
            ):
                try:
                    if args:
                        return method(*args)
                    return method(
                        rollout_id=rollout_id,
                        attempt_id=attempt_id,
                        event=event,
                    )
                except TypeError:
                    continue
        raise AttributeError("Client does not expose an event creation method")

    def enqueue_rollouts(
        self,
        inputs: Iterable[Any],
        step: int,
        *,
        data_ids: Iterable[str] | None = None,
        sample_idx_in_step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[EnqueuedRollout]:
        values = list(inputs)
        ids = list(data_ids) if data_ids is not None else [str(uuid.uuid4()) for _ in values]
        if len(ids) != len(values):
            raise ValueError("data_ids must have the same length as inputs")

        queued: list[EnqueuedRollout] = []
        for offset, (data_id, value) in enumerate(zip(ids, values, strict=True)):
            body: dict[str, Any] = {
                "data_id": str(data_id),
                "input": _to_native(value),
            }
            if self.model_id is not None:
                body["model_id"] = self.model_id
            if metadata:
                body["metadata"] = _to_native(metadata)
            try:
                create = RolloutCreate(**body)
            except Exception:
                create = cast(Any, body)
            rollout = self._create_rollout(create)
            rollout_id = _read_field(rollout, "id") or _read_field(rollout, "rollout_id")
            if rollout_id is None:
                rollout_id = _read_field(create, "id") or str(uuid.uuid4())
            item = EnqueuedRollout(
                data_id=str(data_id),
                rollout_id=str(rollout_id),
                step=step,
                sample_idx_in_step=sample_idx_in_step + offset,
                enqueue_time=time.time(),
                input=value,
                running_at=_read_field(rollout, "running_at"),
                finished_at=_read_field(rollout, "finished_at"),
            )
            self._enqueued_rollouts[item.rollout_id] = item
            queued.append(item)
        return queued

    def enqueue_rollout(
        self,
        input: Any,
        step: int,
        *,
        data_id: str | None = None,
        sample_idx_in_step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> EnqueuedRollout:
        return self.enqueue_rollouts(
            [input],
            step,
            data_ids=[data_id or str(uuid.uuid4())],
            sample_idx_in_step=sample_idx_in_step,
            metadata=metadata,
        )[0]

    def _build_completed_rollout(
        self,
        rollout: Rollout | Mapping[str, Any] | Any,
        enqueued: EnqueuedRollout,
        raw_events: list[Event] | None = None,
    ) -> CompletedRollout:
        events = raw_events if raw_events is not None else self._get_events(enqueued.rollout_id)
        event_dicts = [_dump(event) for event in events]
        triplets: list[Triplet] = []
        triplet_events: list[dict[str, Any]] = []
        final_reward: float | None = None
        rewards_by_attempt: defaultdict[str, list[float]] = defaultdict(list)

        for event, event_dict in zip(events, event_dicts, strict=True):
            typ = _event_type(event)
            data = _event_data(event)
            if typ in {"reward", "rollout_reward", "final_reward"}:
                reward = _as_reward_value(data.get("reward", data.get("value")))
                if reward is not None:
                    final_reward = reward
                    attempt_id = str(_read_field(event, "attempt_id", ""))
                    rewards_by_attempt[attempt_id].append(reward)
                continue
            if typ != "model_request":
                continue

            status = data.get("http_status")
            if data.get("status") == "error" or (isinstance(status, int) and status >= 400):
                continue
            request = data.get("request")
            response_data = data.get("response")
            prompt: Any = request.get("messages") if isinstance(request, Mapping) else request
            response: Any = None
            response_ids: Any = []
            if isinstance(response_data, Mapping):
                choices = response_data.get("choices", [])
                if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                    choice = choices[0]
                    response = choice.get("message", choice.get("text", choice.get("content")))
                    response_ids = choice.get("token_ids", [])
                if response is None:
                    response = response_data.get("output", response_data.get("text"))
            elif isinstance(response_data, list):
                response = response_data
                for chunk in response_data:
                    if isinstance(chunk, Mapping):
                        choices = chunk.get("choices", [])
                        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                            response_ids = choices[0].get("token_ids", response_ids)
            if not response_ids:
                continue

            reward = _as_reward_value(data.get("reward"))
            triplets.append(
                Triplet(
                    prompt=_to_native(prompt),
                    response=_to_native(response),
                    reward=reward,
                    metadata={
                        key: _to_native(value)
                        for key, value in data.items()
                        if key not in {"request", "response", "reward"}
                    },
                )
            )
            triplet_events.append(event_dict)

        images = _aligned_image_urls(cast(list[Event], events), len(triplets))
        if images is not None:
            for triplet, urls in zip(triplets, images, strict=True):
                triplet.image_urls = urls

        rollout_reward = _as_reward_value(
            _read_field(rollout, "reward", _read_field(rollout, "final_reward"))
        )
        if rollout_reward is not None:
            final_reward = rollout_reward
        if final_reward is None and triplets:
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        state = _read_field(rollout, "state")
        error_message = _read_field(rollout, "error_message", _read_field(rollout, "error"))
        return CompletedRollout(
            rollout_id=enqueued.rollout_id,
            data_id=enqueued.data_id,
            step=enqueued.step,
            sample_idx_in_step=enqueued.sample_idx_in_step,
            enqueue_time=enqueued.enqueue_time,
            input=enqueued.input,
            running_at=_read_field(rollout, "running_at", enqueued.running_at),
            finished_at=_read_field(rollout, "finished_at", enqueued.finished_at),
            final_reward=final_reward,
            triplets=triplets or None,
            metadata=_to_native(_read_field(rollout, "metadata", {})) or {},
            events=event_dicts,
            triplet_events=triplet_events,
            rollout_state=state,
            error_message=str(error_message) if error_message is not None else None,
        )

    def get_completed_rollouts(self) -> list[CompletedRollout]:
        completed: list[CompletedRollout] = []
        terminal_values = {
            value.value if hasattr(value, "value") else str(value)
            for value in TERMINAL_STATES
        }
        for rollout_id, enqueued in list(self._enqueued_rollouts.items()):
            if rollout_id in self._completed_rollout_ids:
                continue
            try:
                rollout = self._get_rollout(rollout_id)
            except Exception:
                continue
            state = _read_field(rollout, "state")
            normalized = state.value if hasattr(state, "value") else str(state)
            if normalized not in terminal_values:
                continue
            try:
                result = self._build_completed_rollout(rollout, enqueued)
            except Exception:
                result = CompletedRollout(
                    rollout_id=enqueued.rollout_id,
                    data_id=enqueued.data_id,
                    step=enqueued.step,
                    sample_idx_in_step=enqueued.sample_idx_in_step,
                    enqueue_time=enqueued.enqueue_time,
                    input=enqueued.input,
                    rollout_state=state,
                    error_message=traceback.format_exc(),
                )
            self._completed_rollout_ids.add(rollout_id)
            self._enqueued_rollouts.pop(rollout_id, None)
            completed.append(result)
        return completed

    def poll_completed_rollouts(self) -> list[CompletedRollout]:
        return self.get_completed_rollouts()

    def wait_for_rollouts(
        self,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> list[CompletedRollout]:
        deadline = None if timeout is None else time.monotonic() + timeout
        output: list[CompletedRollout] = []
        while self._enqueued_rollouts:
            output.extend(self.get_completed_rollouts())
            if not self._enqueued_rollouts:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, poll_interval))
        return output

    def flush(self) -> None:
        self._trace_events.flush(self)

    def close(self) -> None:
        self.flush()
        close = getattr(self.client, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "AglRolloutManagerBase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class AglRolloutManager(AglRolloutManagerBase):
    """Default synchronous Agent Lightning rollout manager for VERL."""


__all__ = [
    "AglRolloutManager",
    "AglRolloutManagerBase",
    "CompletedRollout",
    "EnqueuedRollout",
    "Triplet",
]