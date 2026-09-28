from __future__ import annotations

import math
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException

from agentlightning.schemas import DEFAULT_ATTEMPT_ID, Event, EventCreate
from agentlightning.server.store import _events, _rollouts

router = APIRouter(tags=["events"])


def _not_found(rollout_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")


def record_event(
    rollout_id: str,
    attempt_id: str,
    event_type: str,
    data: dict[str, Any],
) -> Event:
    """Append a single event for an existing rollout."""
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    event = Event(
        event_type=event_type,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        timestamp=time.time(),
        data=data,
    )

    rollout_events = _events[rollout_id]
    if attempt_id not in rollout_events:
        rollout_events[attempt_id] = []
    rollout_events[attempt_id].append(event)
    return event


def _query_events(
    rollout_id: str,
    *,
    event_type: str | None = None,
) -> list[Event]:
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    rollout = _rollouts[rollout_id]
    attempt_id = rollout.status.last_attempt_id or DEFAULT_ATTEMPT_ID
    events = _events.get(rollout_id, {}).get(attempt_id, [])

    if event_type is not None:
        events = [event for event in events if event.event_type == event_type]

    return events


def _extract_choice_log_probs(choice: dict[str, Any]) -> list[float] | None:
    """Extract chosen-token logprobs from a single choice."""
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    values: list[Any]
    if isinstance(logprobs.get("content"), list):
        values = []
        for item in logprobs["content"]:
            if not isinstance(item, dict) or "logprob" not in item:
                return None
            values.append(item["logprob"])
    elif isinstance(logprobs.get("token_logprobs"), list):
        values = list(logprobs["token_logprobs"])
    else:
        return None

    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        output.append(number)
    return output


def _trim_model_request(data: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt and response token information from a model request."""
    response = data.get("response")
    prompt_token_ids: list[int] = []
    response_token_ids: list[int] = []
    response_log_probs: list[float] | None = None

    if isinstance(response, dict):
        prompt_token_ids = response.get("prompt_token_ids", [])
        choices = response.get("choices", [])
        if choices:
            if not prompt_token_ids:
                prompt_token_ids = choices[0].get("prompt_token_ids", [])
            response_token_ids = choices[0].get("token_ids", [])
            response_log_probs = _extract_choice_log_probs(choices[0])
    elif isinstance(response, list):
        for chunk in response:
            if not prompt_token_ids and chunk.get("prompt_token_ids"):
                prompt_token_ids = chunk["prompt_token_ids"]
            choices = chunk.get("choices", [])
            if choices:
                token_ids = choices[0].get("token_ids")
                if token_ids:
                    response_token_ids.extend(token_ids)

    server = data.get("server", {})
    trimmed: dict[str, Any] = {
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "response_log_probs": response_log_probs,
        "server": {
            "model": server.get("model"),
            "version": server.get("version"),
        },
    }

    for key in ("http_status", "status"):
        if key in data:
            trimmed[key] = data[key]
    if isinstance(response, dict) and "error" in response:
        trimmed["error"] = response["error"]

    return trimmed


def _trim_reward(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only the scalar value from a reward event."""
    trimmed: dict[str, Any] = {"value": data.get("value")}
    for key in ("source", "reason"):
        if key in data:
            trimmed[key] = data[key]
    return trimmed


def _to_triplet_format(event: Event) -> Event:
    """Trim event data for triplet consumption."""
    if event.event_type == "model_request":
        return event.model_copy(update={"data": _trim_model_request(event.data)})
    if event.event_type == "reward":
        return event.model_copy(update={"data": _trim_reward(event.data)})
    return event


def _dedupe_model_requests_by_prompt_token_ids(events: list[Event]) -> list[Event]:
    """Keep only the latest model request sharing a valid prompt token sequence."""
    last_indexes: dict[tuple[int, ...], int] = {}
    retained_indexes: set[int] = set()

    for index, event in enumerate(events):
        if event.event_type != "model_request":
            continue

        prompt_token_ids = event.data.get("prompt_token_ids", [])
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(type(token_id) is not int for token_id in prompt_token_ids)
        ):
            retained_indexes.add(index)
            continue

        last_indexes[tuple(prompt_token_ids)] = index

    retained_indexes.update(last_indexes.values())
    return [
        event
        for index, event in enumerate(events)
        if event.event_type != "model_request" or index in retained_indexes
    ]


@router.post("/rollouts/{rollout_id}/attempt/{attempt_id}/events", response_model=Event)
async def post_event(rollout_id: str, body: EventCreate, attempt_id: str) -> Event:
    """Post an event for one rollout attempt."""
    return record_event(rollout_id, attempt_id, body.event_type, body.data)


@router.get("/rollouts/{rollout_id}/events", response_model=list[Event])
async def query_events(
    rollout_id: str,
    event_type: str | None = None,
    format: str | None = Query(
        None,
        description="Set to 'triplet' to trim events for RL training",
    ),
) -> list[Event]:
    """Query events for the default rollout attempt."""
    events = _query_events(rollout_id=rollout_id, event_type=event_type)
    if format == "triplet":
        events = [_to_triplet_format(event) for event in events]
        events = _dedupe_model_requests_by_prompt_token_ids(events)
    return events