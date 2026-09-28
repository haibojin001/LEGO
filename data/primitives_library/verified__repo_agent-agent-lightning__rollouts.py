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
    """Rollout together with the identifiers of its attempts."""

    rollout: Rollout
    attempts: list[str]


class TerminalRolloutItem(BaseModel):
    """Minimal terminal-rollout representation."""

    rollout_id: str
    state: RolloutState
    data_id: str
    is_train: bool


class TerminalRolloutsPage(BaseModel):
    """Terminal rollout page and cursor information."""

    items: list[TerminalRolloutItem]
    next_after: int
    total_terminal: int


def _not_found(rollout_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"Rollout not found: {rollout_id}",
    )


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
        attempts.keys(),
        key=lambda attempt_id: (
            attempts[attempt_id][0].timestamp
            if attempts[attempt_id]
            else float("inf")
        ),
    )


@router.post("/rollouts", status_code=201, response_model=list[Rollout])
async def enqueue_rollouts(body: list[RolloutCreate]) -> list[Rollout]:
    """Create queued rollouts, honoring explicitly supplied existing ids."""
    created: list[Rollout] = []

    for request in body:
        if request.rollout_id is not None and request.rollout_id in _rollouts:
            created.append(_rollouts[request.rollout_id])
            continue

        now = time.time()
        rollout_id = request.rollout_id or uuid.uuid4().hex
        rollout = Rollout(
            rollout_id=rollout_id,
            input=request.input,
            is_train=request.is_train,
            config=request.config or RolloutConfig(),
            metadata=_metadata_from_request(request),
            status=RolloutLifecycleStatus(created_at=now, updated_at=now),
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
    """Return rollouts whose current states are among the requested states."""
    wanted_states = set(state_in)
    selected = [
        rollout
        for rollout in _rollouts.values()
        if rollout.status.state in wanted_states
    ]
    return selected[:limit]


def _data_id_of(rollout: Rollout) -> str:
    rollout_input = rollout.input
    if isinstance(rollout_input, dict):
        return str(
            rollout_input.get("data_id")
            or rollout_input.get("instance_id")
            or ""
        )
    return ""


@router.get("/rollouts/terminal", response_model=TerminalRolloutsPage)
async def list_terminal_rollouts(
    after: int = 0,
    limit: int = 1000,
) -> TerminalRolloutsPage:
    """Return terminal rollouts in completion-log order."""
    if after < 0:
        after = 0
    if limit < 1:
        limit = 1

    total_terminal = len(_terminal_order)
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
        total_terminal=total_terminal,
    )


@router.get("/rollouts/{rollout_id}", response_model=RolloutDetail)
async def get_rollout(rollout_id: str) -> RolloutDetail:
    """Fetch one rollout and its known attempts."""
    rollout = _get_rollout(rollout_id)
    return RolloutDetail(
        rollout=rollout,
        attempts=_list_attempts(rollout_id),
    )


@router.patch("/rollouts/{rollout_id}", response_model=Rollout)
async def patch_rollout(rollout_id: str, body: RolloutPatch) -> Rollout:
    """Apply lifecycle status changes to a rollout."""
    rollout = _get_rollout(rollout_id)
    changes = (
        body.status.model_dump(exclude_unset=True)
        if body.status is not None
        else {}
    )

    if not changes:
        return rollout

    if "state" in changes:
        new_state = changes["state"]
        if new_state not in VALID_TRANSITIONS[rollout.status.state]:
            raise _invalid_transition(
                rollout_id,
                rollout.status.state,
                str(new_state),
            )

    next_status = rollout.status.model_copy(
        update={
            **changes,
            "version": rollout.status.version + 1,
            "updated_at": time.time(),
        }
    )
    updated_rollout = rollout.model_copy(update={"status": next_status})
    _rollouts[rollout_id] = updated_rollout

    if "state" in changes and next_status.state in TERMINAL_STATES:
        _terminal_order.append(rollout_id)

    return updated_rollout


@router.delete("/rollouts/{rollout_id}", status_code=204)
async def delete_rollout(rollout_id: str) -> None:
    """Remove a rollout and all of its events, if present."""
    _rollouts.pop(rollout_id, None)
    _events.pop(rollout_id, None)