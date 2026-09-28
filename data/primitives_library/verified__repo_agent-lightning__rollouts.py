from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from agentlightning.schemas import (
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    Rollout,
    RolloutConfig,
    RolloutCreate,
    RolloutLifecycleStatus,
    RolloutMetadata,
    RolloutPatch,
    RolloutState,
)
from agentlightning.server.store import _events, _rollouts, _terminal_order

router = APIRouter(tags=["rollouts"])


class RolloutDetail(BaseModel):
    """Rollout with attempt list."""

    rollout: Rollout
    attempts: list[str]


class TerminalRolloutItem(BaseModel):
    """Lightweight projection of a terminal rollout (no input/config payload)."""

    rollout_id: str
    state: RolloutState
    data_id: str
    is_train: bool


class TerminalRolloutsPage(BaseModel):
    """A page of terminal rollouts plus the cursor to fetch the next page."""

    items: list[TerminalRolloutItem]
    next_after: int
    total_terminal: int


def _not_found(rollout_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")


def _invalid_transition(
    rollout_id: str,
    from_status: str,
    to_status: str,
) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"Rollout {rollout_id}: cannot transition {from_status} -> {to_status}",
    )


def _get_rollout(rollout_id: str) -> Rollout:
    try:
        return _rollouts[rollout_id]
    except KeyError:
        raise _not_found(rollout_id) from None


def _metadata_from_request(req: RolloutCreate) -> RolloutMetadata:
    metadata = req.metadata
    if isinstance(metadata, dict):
        return RolloutMetadata(**metadata)
    if metadata is not None:
        return metadata
    return RolloutMetadata()


def _list_attempts(rollout_id: str) -> list[str]:
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    attempts = _events.get(rollout_id, {})
    if not attempts:
        return []

    return sorted(
        attempts,
        key=lambda attempt_id: (
            attempts[attempt_id][0].timestamp
            if attempts[attempt_id]
            else float("inf")
        ),
    )


@router.post("/rollouts", status_code=201, response_model=list[Rollout])
async def enqueue_rollouts(body: list[RolloutCreate]) -> list[Rollout]:
    """Enqueue rollouts, preserving existing explicitly identified rollouts."""
    created: list[Rollout] = []

    for request in body:
        requested_id = request.rollout_id
        if requested_id is not None and requested_id in _rollouts:
            created.append(_rollouts[requested_id])
            continue

        timestamp = time.time()
        rollout_id = requested_id or uuid.uuid4().hex
        rollout = Rollout(
            rollout_id=rollout_id,
            input=request.input,
            is_train=request.is_train,
            config=request.config or RolloutConfig(),
            metadata=_metadata_from_request(request),
            status=RolloutLifecycleStatus(
                created_at=timestamp,
                updated_at=timestamp,
            ),
        )
        _rollouts[rollout_id] = rollout
        _events[rollout_id] = {}
        created.append(rollout)

    return created


@router.get("/rollouts", response_model=list[Rollout])
async def list_rollouts(
    state_in: Annotated[list[RolloutState], Query()],
    limit: int = 500,
) -> list[Rollout]:
    """List rollouts whose current state is among the supplied states."""
    wanted = set(state_in)
    matching = [
        rollout
        for rollout in _rollouts.values()
        if rollout.status.state in wanted
    ]
    return matching[:limit]


def _data_id_of(rollout: Rollout) -> str:
    value = rollout.input
    if isinstance(value, dict):
        return str(value.get("data_id") or value.get("instance_id") or "")
    return ""


@router.get("/rollouts/terminal", response_model=TerminalRolloutsPage)
async def list_terminal_rollouts(
    after: int = 0,
    limit: int = 1000,
) -> TerminalRolloutsPage:
    """Return terminal rollouts in terminal-transition order."""
    if after < 0:
        after = 0
    if limit < 1:
        limit = 1

    total = len(_terminal_order)
    rollout_ids = _terminal_order[after : after + limit]
    items: list[TerminalRolloutItem] = []

    for rollout_id in rollout_ids:
        rollout = _rollouts.get(rollout_id)
        if rollout is None:
            continue
        items.append(
            TerminalRolloutItem(
                rollout_id=rollout_id,
                state=rollout.status.state,
                data_id=_data_id_of(rollout),
                is_train=rollout.is_train,
            )
        )

    return TerminalRolloutsPage(
        items=items,
        next_after=after + len(rollout_ids),
        total_terminal=total,
    )


@router.get("/rollouts/{rollout_id}", response_model=RolloutDetail)
async def get_rollout(rollout_id: str) -> RolloutDetail:
    """Get a rollout together with its attempt identifiers."""
    rollout = _get_rollout(rollout_id)
    return RolloutDetail(
        rollout=rollout,
        attempts=_list_attempts(rollout_id),
    )


@router.patch("/rollouts/{rollout_id}", response_model=Rollout)
async def patch_rollout(rollout_id: str, body: RolloutPatch) -> Rollout:
    """Patch lifecycle fields on a rollout."""
    rollout = _get_rollout(rollout_id)
    changes = (
        body.status.model_dump(exclude_unset=True)
        if body.status is not None
        else {}
    )

    if not changes:
        return rollout

    if "state" in changes:
        target_state = changes["state"]
        if target_state not in VALID_TRANSITIONS[rollout.status.state]:
            raise _invalid_transition(
                rollout_id,
                rollout.status.state,
                str(target_state),
            )

    status = rollout.status.model_copy(
        update={
            **changes,
            "version": rollout.status.version + 1,
            "updated_at": time.time(),
        }
    )
    updated = rollout.model_copy(update={"status": status})
    _rollouts[rollout_id] = updated

    if "state" in changes and status.state in TERMINAL_STATES:
        _terminal_order.append(rollout_id)

    return updated


@router.delete("/rollouts/{rollout_id}", status_code=204)
async def delete_rollout(rollout_id: str) -> None:
    """Remove a rollout and all of its event attempts."""
    _rollouts.pop(rollout_id, None)
    _events.pop(rollout_id, None)