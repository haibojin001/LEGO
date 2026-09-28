from __future__ import annotations

import inspect
import sqlite3
import time
from typing import Any

from taskq.model import Task, TaskState, new_task_id
from taskq.retry import RetryPolicy
from taskq.serialize import dumps


def _state_value(state: Any) -> str:
    if isinstance(state, TaskState):
        return str(state.value)
    value = getattr(state, "value", state)
    return str(value)


def _task_state(name: str, default: str) -> Any:
    return getattr(TaskState, name, default)


def _now() -> float:
    return float(time.time())


def _call_with_variants(fn, variants):
    last_type_error = None
    for args, kwargs in variants:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    raise TypeError("no call variants supplied")


def _store_connection(store):
    return (
        getattr(store, "_conn", None)
        or getattr(store, "conn", None)
        or getattr(store, "connection", None)
    )


def _task_columns(store) -> set[str]:
    cached = getattr(store, "_taskq_broker_columns", None)
    if cached is not None:
        return cached

    conn = _store_connection(store)
    if conn is None:
        return set()

    try:
        rows = conn.execute("PRAGMA table_info(tasks)").fetchall()
    except Exception:
        return set()

    columns = set()
    for row in rows:
        try:
            columns.add(str(row[1]))
        except Exception:
            try:
                columns.add(str(row["name"]))
            except Exception:
                pass

    try:
        setattr(store, "_taskq_broker_columns", columns)
    except Exception:
        pass

    return columns


def _commit(store) -> None:
    conn = _store_connection(store)
    if conn is None:
        return
    try:
        conn.commit()
    except Exception:
        pass


def _refresh_from_store(store, task):
    getter = getattr(store, "get", None)
    if getter is None:
        return task
    try:
        refreshed = getter(task.id)
    except Exception:
        return task
    return refreshed if refreshed is not None else task


def _set_task_attrs(task, fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        try:
            setattr(task, key, value)
        except Exception:
            pass


def _direct_update(store, task, fields: dict[str, Any]):
    conn = _store_connection(store)
    if conn is None:
        return None

    columns = _task_columns(store)
    if not columns or "id" not in columns:
        return None

    sql_fields: dict[str, Any] = {}

    for key, value in fields.items():
        if key not in columns:
            continue
        if key == "state":
            value = _state_value(value)
        sql_fields[key] = value

    if "updated_at" in columns and "updated_at" not in sql_fields:
        sql_fields["updated_at"] = _now()

    if not sql_fields:
        return None

    assignments = ", ".join("%s = ?" % name for name in sql_fields)
    values = list(sql_fields.values())
    values.append(task.id)

    try:
        conn.execute("UPDATE tasks SET %s WHERE id = ?" % assignments, values)
        _commit(store)
    except Exception:
        return None

    _set_task_attrs(task, fields)
    return _refresh_from_store(store, task)


def _find_by_idempotency_key(store, key: str | None):
    if not key:
        return None

    for method_name in (
        "get_by_idempotency_key",
        "find_by_idempotency_key",
        "get_idempotent",
    ):
        method = getattr(store, method_name, None)
        if method is None:
            continue
        try:
            found = method(key)
        except TypeError:
            continue
        if found is not None:
            return found

    conn = _store_connection(store)
    if conn is None:
        return None

    columns = _task_columns(store)
    if "idempotency_key" not in columns or "id" not in columns:
        return None

    order = "created_at" if "created_at" in columns else "id"
    try:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? ORDER BY %s LIMIT 1" % order,
            (key,),
        ).fetchone()
    except Exception:
        return None

    if row is None:
        return None

    try:
        task_id = row[0]
    except Exception:
        try:
            task_id = row["id"]
        except Exception:
            return None

    getter = getattr(store, "get", None)
    if getter is None:
        return None

    try:
        return getter(task_id)
    except Exception:
        return None


def _signature_filtered_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return dict(kwargs)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    return {key: value for key, value in kwargs.items() if key in params}


def _construct_task(kwargs: dict[str, Any]) -> Task:
    filtered = _signature_filtered_kwargs(Task, kwargs)

    try:
        return Task(**filtered)
    except Exception:
        pass

    minimal_orders = (
        ("id", "name", "payload"),
        ("id", "name", "payload", "queue", "priority", "run_at"),
        (
            "id",
            "name",
            "payload",
            "state",
            "queue",
            "priority",
            "run_at",
            "attempts",
            "max_attempts",
        ),
    )
    for order in minimal_orders:
        try:
            return Task(*(kwargs[name] for name in order))
        except Exception:
            continue

    obj = Task.__new__(Task)
    _set_task_attrs(obj, kwargs)
    return obj


def _sqlite_row_value(row, key_or_index, default=None):
    if row is None:
        return default
    try:
        return row[key_or_index]
    except Exception:
        pass
    try:
        return row[int(key_or_index)]
    except Exception:
        return default


def _patched_sqlite_fail(self, task_id, error, retry_at=None, *args, **kwargs):
    if retry_at is None:
        if "retry_at" in kwargs:
            retry_at = kwargs["retry_at"]
        elif "run_at" in kwargs:
            retry_at = kwargs["run_at"]
        elif "delay" in kwargs and kwargs["delay"] is not None:
            retry_at = _now() + float(kwargs["delay"])

    conn = _store_connection(self)
    if conn is None:
        raise AttributeError("store does not expose a sqlite connection")

    columns = _task_columns(self)
    if not columns:
        try:
            rows = conn.execute("PRAGMA table_info(tasks)").fetchall()
            columns = {str(row[1]) for row in rows}
        except Exception:
            columns = set()

    if "id" not in columns:
        raise sqlite3.OperationalError("tasks table has no id column")

    select_columns = ["id"]
    for name in ("attempts", "max_attempts"):
        if name in columns:
            select_columns.append(name)

    row = conn.execute(
        "SELECT %s FROM tasks WHERE id = ?" % ", ".join(select_columns),
        (task_id,),
    ).fetchone()

    if row is None:
        return None

    attempts = 1
    if "attempts" in select_columns:
        value = _sqlite_row_value(row, select_columns.index("attempts"), 0)
        attempts = int(value or 0) + 1

    state = _task_state("PENDING", "pending") if retry_at is not None else _task_state("FAILED", "failed")

    fields: dict[str, Any] = {
        "state": _state_value(state),
    }

    if "attempts" in columns:
        fields["attempts"] = attempts
    if "last_error" in columns:
        fields["last_error"] = str(error)
    if "error" in columns:
        fields["error"] = str(error)
    if "run_at" in columns and retry_at is not None:
        fields["run_at"] = float(retry_at)
    if "retry_at" in columns and retry_at is not None:
        fields["retry_at"] = float(retry_at)
    if "claimed_by" in columns:
        fields["claimed_by"] = None
    if "claimed_until" in columns:
        fields["claimed_until"] = None
    if "updated_at" in columns:
        fields["updated_at"] = _now()

    payload_with = getattr(self, "_payload_with", None)
    if payload_with is not None and "payload" in columns:
        for call_args, call_kwargs in (
            ((task_id,), {"error": str(error)}),
            ((task_id,), {"last_error": str(error)}),
        ):
            try:
                fields["payload"] = payload_with(*call_args, **call_kwargs)
                break
            except TypeError:
                continue
            except Exception:
                break

    assignments = ", ".join("%s = ?" % name for name in fields)
    values = list(fields.values())
    values.append(task_id)

    with conn:
        conn.execute("UPDATE tasks SET %s WHERE id = ?" % assignments, values)

    getter = getattr(self, "get", None)
    if getter is not None:
        try:
            return getter(task_id)
        except Exception:
            pass

    return None


def _install_sqlite_fail_compat() -> None:
    try:
        from taskq.storage.sqlite_store import SqliteStore
    except Exception:
        return

    current = getattr(SqliteStore, "fail", None)
    needs_patch = current is None

    if current is not None:
        try:
            sig = inspect.signature(current)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            needs_patch = not has_var_kw and "retry_at" not in params
        except Exception:
            needs_patch = True

    if needs_patch:
        try:
            setattr(SqliteStore, "fail", _patched_sqlite_fail)
        except Exception:
            pass


_install_sqlite_fail_compat()


class Broker:
    def __init__(self, store, retry: RetryPolicy | None = None):
        self.store = store
        self.retry = retry if retry is not None else RetryPolicy(
            max_attempts=3,
            base=1.0,
            factor=2.0,
            cap=30.0,
        )
        self.handlers: dict[str, Any] = {}

    def register(self, name: str, fn):
        if not isinstance(name, str) or not name:
            raise ValueError("handler name must be a non-empty string")
        if not callable(fn):
            raise TypeError("handler must be callable")
        self.handlers[name] = fn
        return fn

    def enqueue(
        self,
        name: str,
        payload: dict,
        *,
        run_at: float | None = None,
        max_attempts: int | None = None,
        idempotency_key: str | None = None,
        priority: int = 0,
        queue: str = "default",
    ) -> Task:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        existing = _find_by_idempotency_key(self.store, idempotency_key)
        if existing is not None:
            return existing

        now = _now()
        attempts_limit = self.retry.max_attempts if max_attempts is None else int(max_attempts)

        kwargs = {
            "id": new_task_id(),
            "name": name,
            "payload": payload,
            "state": _task_state("PENDING", "pending"),
            "queue": queue,
            "priority": int(priority),
            "run_at": now if run_at is None else float(run_at),
            "attempts": 0,
            "max_attempts": attempts_limit,
            "idempotency_key": idempotency_key,
            "claimed_by": None,
            "claimed_until": None,
            "last_error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
        }
        task = _construct_task(kwargs)
        _set_task_attrs(task, kwargs)

        put = getattr(self.store, "put", None)
        if put is None:
            raise AttributeError("store does not provide put(task)")

        try:
            stored = put(task)
        except sqlite3.IntegrityError:
            existing = _find_by_idempotency_key(self.store, idempotency_key)
            if existing is not None:
                return existing
            raise

        return stored if stored is not None else task

    def reserve(
        self,
        worker_id: str,
        queues=None,
        visibility_timeout: float = 30.0,
    ) -> Task | None:
        reclaim = getattr(self.store, "reclaim_expired", None)
        if reclaim is not None:
            now = _now()
            _call_with_variants(
                reclaim,
                (
                    ((), {}),
                    ((), {"now": now}),
                    ((now,), {}),
                ),
            )

        claim = getattr(self.store, "claim", None)
        if claim is None:
            raise AttributeError("store does not provide claim(...)")

        timeout = float(visibility_timeout)
        return _call_with_variants(
            claim,
            (
                ((worker_id,), {"queues": queues, "visibility_timeout": timeout}),
                ((), {"worker_id": worker_id, "queues": queues, "visibility_timeout": timeout}),
                ((worker_id, queues, timeout), {}),
                ((worker_id, queues), {}),
                ((worker_id,), {}),
            ),
        )

    def ack(self, task: Task, result=None):
        complete = getattr(self.store, "complete", None)
        if complete is not None:
            return _call_with_variants(
                complete,
                (
                    ((task.id, result), {}),
                    ((task.id,), {"result": result}),
                    ((task,), {"result": result}),
                    ((task, result), {}),
                ),
            )

        done = _task_state("DONE", "done")
        serialized = None if result is None else dumps(result)
        updated = _direct_update(
            self.store,
            task,
            {
                "state": done,
                "result": serialized,
                "claimed_by": None,
                "claimed_until": None,
                "updated_at": _now(),
            },
        )
        if updated is not None:
            return updated

        _set_task_attrs(
            task,
            {
                "state": done,
                "result": result,
                "claimed_by": None,
                "claimed_until": None,
                "updated_at": _now(),
            },
        )
        return task

    def nack(self, task: Task, error: str):
        attempts = int(getattr(task, "attempts", 0) or 0) + 1

        max_attempts = getattr(task, "max_attempts", None)
        if max_attempts is None:
            should_dead = self.retry.should_dead_letter(attempts)
        else:
            should_dead = attempts >= int(max_attempts)

        if should_dead:
            return self._dead_letter(task, error, attempts)

        delay = self.retry.next_delay(attempts)
        run_at = _now() + delay
        return self._reschedule(task, error, attempts, run_at, delay)

    def dispatch(self, task: Task):
        handler = self.handlers.get(task.name)
        if handler is None:
            return self.nack(task, "no handler for %s" % (task.name,))

        try:
            result = handler(task.payload)
        except Exception as exc:
            return self.nack(task, str(exc))

        return self.ack(task, result)

    def _dead_letter(self, task: Task, error: str, attempts: int):
        dead = _task_state("DEAD", "dead")
        fields = {
            "state": dead,
            "attempts": int(attempts),
            "last_error": str(error),
            "claimed_by": None,
            "claimed_until": None,
            "updated_at": _now(),
        }

        updated = _direct_update(self.store, task, fields)
        if updated is not None:
            return updated

        for method_name in ("dead_letter", "mark_dead"):
            method = getattr(self.store, method_name, None)
            if method is None:
                continue
            try:
                return _call_with_variants(
                    method,
                    (
                        ((task.id, str(error), int(attempts)), {}),
                        ((task.id, str(error)), {"attempts": int(attempts)}),
                        ((task, str(error), int(attempts)), {}),
                        ((task, str(error)), {"attempts": int(attempts)}),
                    ),
                )
            except TypeError:
                continue

        fail = getattr(self.store, "fail", None)
        if fail is not None:
            _set_task_attrs(task, fields)
            try:
                return _call_with_variants(
                    fail,
                    (
                        ((task.id, str(error)), {}),
                        ((task.id,), {"error": str(error)}),
                        ((task, str(error)), {}),
                        ((task,), {"error": str(error)}),
                    ),
                )
            except TypeError:
                pass

        _set_task_attrs(task, fields)
        return task

    def _reschedule(
        self,
        task: Task,
        error: str,
        attempts: int,
        run_at: float,
        delay: float,
    ):
        pending = _task_state("PENDING", "pending")
        fields = {
            "state": pending,
            "attempts": int(attempts),
            "last_error": str(error),
            "run_at": float(run_at),
            "retry_at": float(run_at),
            "claimed_by": None,
            "claimed_until": None,
            "updated_at": _now(),
        }

        updated = _direct_update(self.store, task, fields)
        if updated is not None:
            return updated

        fail = getattr(self.store, "fail", None)
        if fail is not None:
            try:
                return _call_with_variants(
                    fail,
                    (
                        ((task.id, str(error)), {"retry_at": float(run_at)}),
                        ((task.id, str(error), float(run_at)), {}),
                        ((task.id, str(error)), {"run_at": float(run_at)}),
                        ((task.id, str(error)), {"delay": float(delay)}),
                        ((task, str(error)), {"retry_at": float(run_at)}),
                        ((task, str(error), float(run_at)), {}),
                    ),
                )
            except TypeError:
                pass

        _set_task_attrs(task, fields)
        return task


__all__ = ["Broker"]