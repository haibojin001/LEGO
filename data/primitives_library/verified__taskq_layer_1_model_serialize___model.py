from enum import Enum
import os
import secrets
import threading
import time


class TaskState(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


_TERMINAL_STATES = {TaskState.DONE, TaskState.FAILED, TaskState.DEAD}


def _coerce_state(state):
    if isinstance(state, TaskState):
        return state
    return TaskState(state)


class Task:
    def __init__(
        self,
        id,
        name,
        payload,
        state="pending",
        attempts=0,
        max_attempts=3,
        run_at=0.0,
        claimed_by=None,
        claim_expires=0.0,
        last_error="",
        idempotency_key=None,
        result=None,
        priority=0,
        queue="default",
    ):
        self.id = id
        self.name = name
        self.payload = {} if payload is None else payload
        self.state = _coerce_state(state)
        self.attempts = int(attempts)
        self.max_attempts = int(max_attempts)
        self.run_at = float(run_at)
        self.claimed_by = claimed_by
        self.claim_expires = float(claim_expires)
        self.last_error = last_error
        self.idempotency_key = idempotency_key
        self.result = result
        self.priority = int(priority)
        self.queue = queue

    def to_dict(self) -> dict:
        state = _coerce_state(self.state).value
        return {
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
            "state": state,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "run_at": self.run_at,
            "claimed_by": self.claimed_by,
            "claim_expires": self.claim_expires,
            "last_error": self.last_error,
            "idempotency_key": self.idempotency_key,
            "result": self.result,
            "priority": self.priority,
            "queue": self.queue,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            d["id"],
            d["name"],
            d.get("payload", {}),
            d.get("state", TaskState.PENDING.value),
            d.get("attempts", 0),
            d.get("max_attempts", 3),
            d.get("run_at", 0.0),
            d.get("claimed_by", None),
            d.get("claim_expires", 0.0),
            d.get("last_error", ""),
            d.get("idempotency_key", None),
            d.get("result", None),
            d.get("priority", 0),
            d.get("queue", "default"),
        )

    def is_terminal(self) -> bool:
        return _coerce_state(self.state) in _TERMINAL_STATES


_id_lock = threading.Lock()
_id_last_ns = 0
_id_seq = 0
_id_salt = secrets.token_hex(4)


def new_task_id() -> str:
    global _id_last_ns, _id_seq

    with _id_lock:
        now_ns = time.time_ns()
        if now_ns <= _id_last_ns:
            now_ns = _id_last_ns + 1
        _id_last_ns = now_ns
        _id_seq = (_id_seq + 1) & 0xFFFFFFFF
        seq = _id_seq

    pid = os.getpid() & 0xFFFFFFFF
    return f"{now_ns:020d}{pid:08x}{_id_salt}{seq:08x}"