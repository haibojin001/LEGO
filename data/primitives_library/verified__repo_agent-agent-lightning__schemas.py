from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    event_type: str
    rollout_id: str
    attempt_id: str
    timestamp: float
    data: dict[str, Any]


class EventCreate(BaseModel):
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ModelRequestData(BaseModel):
    model: str
    model_version: int | None = None
    request: dict[str, Any]
    adjusted_params: dict[str, Any] | None = None
    response: dict[str, Any]
    latency_ms: float | None = None
    http_status: int | None = None
    status: str = "ok"
    retry_count: int = 0
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class RewardData(BaseModel):
    value: float
    message: str | None = None
    source: str | None = None
    reason: str | None = None


class Model(BaseModel):
    model: str
    endpoint: str
    version: int = 0


class RolloutState(StrEnum):
    QUEUING = "queuing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


VALID_TRANSITIONS: dict[RolloutState, set[RolloutState]] = {
    RolloutState.QUEUING: {RolloutState.RUNNING, RolloutState.FAILED},
    RolloutState.RUNNING: {RolloutState.SUCCEEDED, RolloutState.FAILED},
    RolloutState.SUCCEEDED: set(),
    RolloutState.FAILED: set(),
}

TERMINAL_STATES: frozenset[RolloutState] = frozenset(
    {RolloutState.SUCCEEDED, RolloutState.FAILED}
)

DEFAULT_ATTEMPT_ID = "0"


class RolloutLocalConfig(BaseModel):
    agent_class: str | None = None
    env_map: dict[str, str] = Field(default_factory=dict)


class RolloutK8sConfig(BaseModel):
    job_template: str | None = None


class RolloutConfig(BaseModel):
    timeout_seconds: int = 3600
    local: RolloutLocalConfig | None = None
    k8s: RolloutK8sConfig | None = None


class RolloutMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    batch_idx: int | None = None
    sample_idx_in_batch: int | None = None


class RolloutCreate(BaseModel):
    input: Any
    is_train: bool = True
    config: RolloutConfig | None = None
    metadata: RolloutMetadata | dict[str, Any] | None = None
    rollout_id: str | None = None


class RolloutLifecycleStatus(BaseModel):
    state: RolloutState = RolloutState.QUEUING
    k8s_job_name: str | None = None
    last_attempt_id: str | None = None
    error_message: str | None = None
    version: int = 1
    created_at: float
    updated_at: float


class RolloutStatusPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: RolloutState | None = None
    k8s_job_name: str | None = None
    last_attempt_id: str | None = None
    error_message: str | None = None


class RolloutPatch(BaseModel):
    status: RolloutStatusPatch | None = None


class Rollout(BaseModel):
    rollout_id: str
    input: Any
    is_train: bool = True
    config: RolloutConfig
    metadata: RolloutMetadata = Field(default_factory=RolloutMetadata)
    status: RolloutLifecycleStatus