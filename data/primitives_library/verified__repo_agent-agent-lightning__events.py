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


def record_event(rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> Event:
    """Append a single event for an existing rollout."""
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    created = Event(
        event_type=event_type,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        timestamp=time.time(),
        data=data,
    )

    by_attempt = _events[rollout_id]
    if attempt_id not in by_attempt:
        by_attempt[attempt_id] = []
    by_attempt[attempt_id].append(created)
    return created


def _query_events(
    rollout_id: str,
    *,
    event_type: str | None = None,
) -> list[Event]:
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    attempt_id = _rollouts[rollout_id].status.last_attempt_id or DEFAULT_ATTEMPT_ID
    found = _events.get(rollout_id, {}).get(attempt_id, [])
    if event_type is not None:
        found = [item for item in found if item.event_type == event_type]
    return found


def _extract_choice_log_probs(choice: dict[str, Any]) -> list[float] | None:
    """Return valid selected-token log probabilities for a response choice."""
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    values: list[Any]
    content = logprobs.get("content")
    token_logprobs = logprobs.get("token_logprobs")

    if isinstance(content, list):
        values = []
        for entry in content:
            if not isinstance(entry, dict) or "logprob" not in entry:
                return None
            values.append(entry["logprob"])
    elif isinstance(token_logprobs, list):
        values = list(token_logprobs)
    else:
        return None

    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        converted.append(number)
    return converted


def _trim_model_request(data: dict[str, Any]) -> dict[str, Any]:
    """Extract the subset of model request data used by triplet training."""
    response = data.get("response")
    prompt_token_ids: list[int] = []
    response_token_ids: list[int] = []
    response_log_probs: list[float] | None = None

    if isinstance(response, dict):
        prompt_token_ids = response.get("prompt_token_ids", [])
        choices = response.get("choices", [])
        if choices:
            first_choice = choices[0]
            if not prompt_token_ids:
                prompt_token_ids = first_choice.get("prompt_token_ids", [])
            response_token_ids = first_choice.get("token_ids", [])
            response_log_probs = _extract_choice_log_probs(first_choice)
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
    result = {
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "response_log_probs": response_log_probs,
        "server": {
            "model": server.get("model"),
            "version": server.get("version"),
        },
    }

    for field in ("http_status", "status"):
        if field in data:
            result[field] = data[field]

    if isinstance(response, dict) and "error" in response:
        result["error"] = response["error"]

    return result


def _trim_reward(data: dict[str, Any]) -> dict[str, Any]:
    """Retain the reward value and optional provenance fields."""
    result = {"value": data.get("value")}
    for field in ("source", "reason"):
        if field in data:
            result[field] = data[field]
    return result


def _to_triplet_format(event: Event) -> Event:
    """Convert supported event payloads into their triplet representation."""
    if event.event_type == "model_request":
        return event.model_copy(update={"data": _trim_model_request(event.data)})
    if event.event_type == "reward":
        return event.model_copy(update={"data": _trim_reward(event.data)})
    return event


def _dedupe_model_requests_by_prompt_token_ids(events: list[Event]) -> list[Event]:
    """For each valid prompt token sequence, retain only its final request."""
    final_positions: dict[tuple[int, ...], int] = {}
    retained: set[int] = set()

    for position, event in enumerate(events):
        if event.event_type != "model_request":
            continue

        tokens = event.data.get("prompt_token_ids", [])
        valid_key = (
            isinstance(tokens, list)
            and bool(tokens)
            and not any(type(token) is not int for token in tokens)
        )
        if not valid_key:
            retained.add(position)
            continue

        final_positions[tuple(tokens)] = position

    retained.update(final_positions.values())
    return [
        event
        for position, event in enumerate(events)
        if event.event_type != "model_request" or position in retained
    ]


@router.post("/rollouts/{rollout_id}/attempt/{attempt_id}/events", response_model=Event)
async def post_event(rollout_id: str, body: EventCreate, attempt_id: str) -> Event:
    """Post an event for one rollout attempt."""
    return record_event(rollout_id, attempt_id, body.event_type, body.data)


@router.get("/rollouts/{rollout_id}/events", response_model=list[Event])
async def query_events(
    rollout_id: str,
    event_type: str | None = None,
    format: str | None = Query(None, description="Set to 'triplet' to trim events for RL training"),
) -> list[Event]:
    """Query events for the default rollout attempt."""
    result = _query_events(rollout_id=rollout_id, event_type=event_type)
    if format == "triplet":
        result = [_to_triplet_format(event) for event in result]
        result = _dedupe_model_requests_by_prompt_token_ids(result)
    return result