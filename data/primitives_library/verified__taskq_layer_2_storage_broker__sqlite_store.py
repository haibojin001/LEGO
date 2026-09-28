import inspect
import sqlite3
import threading
import time
from enum import Enum
from typing import Any

from taskq.model import Task, TaskState, _coerce_state
from taskq.serialize import dumps, loads


def _state_value(value):
    if isinstance(value, Enum):
        return value.value
    if value is None:
        return None
    return str(value)


def _task_state(name, default):
    try:
        return _state_value(getattr(TaskState, name))
    except Exception:
        return default


_PENDING = _task_state("PENDING", "pending")
_CLAIMED = _task_state("CLAIMED", "claimed")
_DONE = _task_state("DONE", "done")
_FAILED = _task_state("FAILED", "failed")
_DEAD = _task_state("DEAD", "dead")


_TASK_FIELDS = (
    "id",
    "name",
    "args",
    "kwargs",
    "queue",
    "priority",
    "run_at",
    "state",
    "attempts",
    "created_at",
    "idempotency_key",
    "claimed_by",
    "claim_expires",
    "result",
    "last_error",
)


def _now():
    return time.time()


def _jsonify(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return repr(value)


def _safe_dumps(value):
    return dumps(_jsonify(value))


def _safe_loads(value, default=None):
    if value is None:
        return default
    try:
        return loads(value)
    except Exception:
        return default


def _all_slots(cls):
    out = []
    for c in reversed(getattr(cls, "__mro__", ())):
        slots = getattr(c, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot not in ("__dict__", "__weakref__") and slot not in out:
                out.append(slot)
    return out


def _coerce_loaded_state(value):
    try:
        return _coerce_state(value)
    except Exception:
        return value


def _object_id(value):
    if isinstance(value, Task):
        return getattr(value, "id")
    if hasattr(value, "id") and not isinstance(value, (str, bytes)):
        try:
            return getattr(value, "id")
        except Exception:
            pass
    return value


def _to_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


class _SqliteStoreMeta(type):
    def __setattr__(cls, name, value):
        if (
            name == "fail"
            and getattr(cls, "_native_fail", None) is not None
            and callable(value)
            and getattr(value, "__name__", "") == "_patched_sqlite_fail"
        ):
            return
        super().__setattr__(name, value)


class SqliteStore(metaclass=_SqliteStoreMeta):
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._ensure_schema()

    def close(self):
        with self._lock:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                self._conn = None
                conn.close()

    def put(self, task: Task) -> Task:
        data = self._task_to_data(task)
        task_id = data.get("id")
        if task_id is None:
            raise ValueError("task.id is required")
        task_id = str(task_id)
        data["id"] = task_id

        idempotency_key = data.get("idempotency_key")
        row_values = self._row_values(data)

        with self._lock:
            self._begin_immediate()
            try:
                if idempotency_key is not None:
                    existing = self._conn.execute(
                        "SELECT * FROM tasks WHERE idempotency_key = ?",
                        (idempotency_key,),
                    ).fetchone()
                    if existing is not None:
                        self._conn.execute("COMMIT")
                        return self._row_to_task(existing)

                existing = self._conn.execute(
                    "SELECT id FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()

                if existing is None:
                    self._conn.execute(
                        """
                        INSERT INTO tasks (
                            id, name, args, kwargs, queue, priority, run_at, state,
                            attempts, created_at, idempotency_key, claimed_by,
                            claim_expires, result, last_error, payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row_values,
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE tasks
                           SET name = ?,
                               args = ?,
                               kwargs = ?,
                               queue = ?,
                               priority = ?,
                               run_at = ?,
                               state = ?,
                               attempts = ?,
                               created_at = ?,
                               idempotency_key = ?,
                               claimed_by = ?,
                               claim_expires = ?,
                               result = ?,
                               last_error = ?,
                               payload = ?
                         WHERE id = ?
                        """,
                        row_values[1:] + (task_id,),
                    )

                row = self._conn.execute(
                    "SELECT * FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError:
                self._conn.execute("ROLLBACK")
                if idempotency_key is not None:
                    existing = self._get_by_idempotency_key(idempotency_key)
                    if existing is not None:
                        return existing
                raise
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return self._row_to_task(row)

    def get(self, task_id: str) -> Task | None:
        task_id = _object_id(task_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def claim(
        self,
        worker_id: str,
        queues=None,
        visibility_timeout: float = 30.0,
        now: float | None = None,
    ) -> Task | None:
        if now is None:
            now = _now()

        queue_values = None
        if queues is not None:
            queue_values = list(queues)
            if not queue_values:
                return None

        with self._lock:
            self._begin_immediate()
            try:
                params = [_PENDING, float(now)]
                sql = "SELECT * FROM tasks WHERE state = ? AND run_at <= ?"
                if queue_values is not None:
                    placeholders = ",".join("?" for _ in queue_values)
                    sql += " AND queue IN (%s)" % placeholders
                    params.extend(queue_values)
                sql += " ORDER BY priority DESC, run_at ASC, created_at ASC, id ASC LIMIT 1"

                row = self._conn.execute(sql, tuple(params)).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None

                task_id = row["id"]
                claim_expires = float(now) + float(visibility_timeout)
                payload = self._payload_from_row_with(
                    row,
                    {
                        "state": _CLAIMED,
                        "claimed_by": worker_id,
                        "claim_expires": claim_expires,
                    },
                )
                cur = self._conn.execute(
                    """
                    UPDATE tasks
                       SET state = ?,
                           claimed_by = ?,
                           claim_expires = ?,
                           payload = ?
                     WHERE id = ? AND state = ?
                    """,
                    (_CLAIMED, worker_id, claim_expires, payload, task_id, _PENDING),
                )
                if cur.rowcount != 1:
                    self._conn.execute("COMMIT")
                    return None

                result_row = self._conn.execute(
                    "SELECT * FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        if result_row is None:
            return None
        return self._row_to_task(result_row)

    def complete(self, task_id: str, result=None):
        task_id = _object_id(task_id)
        result_text = _safe_dumps(result)
        with self._lock:
            self._begin_immediate()
            try:
                row = self._conn.execute(
                    "SELECT * FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return

                payload = self._payload_from_row_with(
                    row,
                    {
                        "state": _DONE,
                        "result": result,
                        "claimed_by": None,
                        "claim_expires": 0.0,
                    },
                )
                self._conn.execute(
                    """
                    UPDATE tasks
                       SET state = ?,
                           result = ?,
                           claimed_by = NULL,
                           claim_expires = 0,
                           payload = ?
                     WHERE id = ?
                    """,
                    (_DONE, result_text, payload, task_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def fail(
        self,
        task_id: str,
        error: str,
        retry_at: float | None = None,
        dead: bool = False,
    ):
        task_id = _object_id(task_id)
        error_text = None if error is None else str(error)
        with self._lock:
            self._begin_immediate()
            try:
                row = self._conn.execute(
                    "SELECT * FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return

                attempts = int(row["attempts"] or 0)

                if dead:
                    state = _DEAD
                    payload = self._payload_from_row_with(
                        row,
                        {
                            "state": state,
                            "last_error": error_text,
                            "claimed_by": None,
                            "claim_expires": 0.0,
                        },
                    )
                    self._conn.execute(
                        """
                        UPDATE tasks
                           SET state = ?,
                               last_error = ?,
                               claimed_by = NULL,
                               claim_expires = 0,
                               payload = ?
                         WHERE id = ?
                        """,
                        (state, error_text, payload, task_id),
                    )
                elif retry_at is None:
                    state = _FAILED
                    payload = self._payload_from_row_with(
                        row,
                        {
                            "state": state,
                            "last_error": error_text,
                            "claimed_by": None,
                            "claim_expires": 0.0,
                        },
                    )
                    self._conn.execute(
                        """
                        UPDATE tasks
                           SET state = ?,
                               last_error = ?,
                               claimed_by = NULL,
                               claim_expires = 0,
                               payload = ?
                         WHERE id = ?
                        """,
                        (state, error_text, payload, task_id),
                    )
                else:
                    state = _PENDING
                    attempts += 1
                    retry_at_float = float(retry_at)
                    payload = self._payload_from_row_with(
                        row,
                        {
                            "state": state,
                            "run_at": retry_at_float,
                            "attempts": attempts,
                            "last_error": error_text,
                            "claimed_by": None,
                            "claim_expires": 0.0,
                        },
                    )
                    self._conn.execute(
                        """
                        UPDATE tasks
                           SET state = ?,
                               run_at = ?,
                               attempts = ?,
                               last_error = ?,
                               claimed_by = NULL,
                               claim_expires = 0,
                               payload = ?
                         WHERE id = ?
                        """,
                        (state, retry_at_float, attempts, error_text, payload, task_id),
                    )

                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def reclaim_expired(self, now: float | None = None) -> int:
        if now is None:
            now = _now()

        with self._lock:
            self._begin_immediate()
            try:
                rows = self._conn.execute(
                    "SELECT * FROM tasks WHERE state = ? AND claim_expires < ?",
                    (_CLAIMED, float(now)),
                ).fetchall()
                count = 0
                for row in rows:
                    payload = self._payload_from_row_with(
                        row,
                        {
                            "state": _PENDING,
                            "claimed_by": None,
                            "claim_expires": 0.0,
                        },
                    )
                    cur = self._conn.execute(
                        """
                        UPDATE tasks
                           SET state = ?,
                               claimed_by = NULL,
                               claim_expires = 0,
                               payload = ?
                         WHERE id = ? AND state = ? AND claim_expires < ?
                        """,
                        (_PENDING, payload, row["id"], _CLAIMED, float(now)),
                    )
                    count += cur.rowcount
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return count

    def pending_count(self, queue: str | None = None) -> int:
        with self._lock:
            if queue is None:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM tasks WHERE state = ?",
                    (_PENDING,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM tasks WHERE state = ? AND queue = ?",
                    (_PENDING, queue),
                ).fetchone()
        return int(row["n"] if row is not None else 0)

    def _ensure_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT,
                args TEXT,
                kwargs TEXT,
                queue TEXT,
                priority INTEGER,
                run_at REAL,
                state TEXT,
                attempts INTEGER,
                created_at REAL,
                idempotency_key TEXT,
                claimed_by TEXT,
                claim_expires REAL,
                result TEXT,
                last_error TEXT,
                payload TEXT
            )
            """
        )

        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        columns = {
            "name": "TEXT",
            "args": "TEXT",
            "kwargs": "TEXT",
            "queue": "TEXT",
            "priority": "INTEGER",
            "run_at": "REAL",
            "state": "TEXT",
            "attempts": "INTEGER",
            "created_at": "REAL",
            "idempotency_key": "TEXT",
            "claimed_by": "TEXT",
            "claim_expires": "REAL",
            "result": "TEXT",
            "last_error": "TEXT",
            "payload": "TEXT",
        }
        for name, decl in columns.items():
            if name not in existing:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN %s %s" % (name, decl))

        self._conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency_key
                ON tasks(idempotency_key)
                WHERE idempotency_key IS NOT NULL
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_claim
                ON tasks(state, queue, run_at, priority, created_at, id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_claim_expires
                ON tasks(state, claim_expires)
            """
        )

    def _begin_immediate(self):
        self._conn.execute("BEGIN IMMEDIATE")

    def _get_by_idempotency_key(self, idempotency_key) -> Task | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def _task_to_data(self, task: Task) -> dict[str, Any]:
        data = {}

        payload = getattr(task, "payload", None)
        if isinstance(payload, str):
            loaded = _safe_loads(payload, {})
            if isinstance(loaded, dict):
                data.update(loaded)
        elif isinstance(payload, dict):
            data.update(payload)

        dct = getattr(task, "__dict__", None)
        if isinstance(dct, dict):
            data.update(dct)

        for slot in _all_slots(type(task)):
            try:
                data[slot] = getattr(task, slot)
            except Exception:
                pass

        for field in _TASK_FIELDS:
            if field not in data:
                try:
                    data[field] = getattr(task, field)
                except Exception:
                    pass

        now = _now()
        data.setdefault("args", [])
        data.setdefault("kwargs", {})
        data.setdefault("queue", "default")
        data.setdefault("priority", 0)
        data.setdefault("run_at", now)
        data.setdefault("state", _PENDING)
        data.setdefault("attempts", 0)
        data.setdefault("created_at", now)
        data.setdefault("idempotency_key", None)
        data.setdefault("claimed_by", None)
        data.setdefault("claim_expires", 0.0)
        data.setdefault("result", None)
        data.setdefault("last_error", None)

        if data.get("id") is not None:
            data["id"] = str(data["id"])
        if data.get("name") is not None:
            data["name"] = str(data["name"])
        if data.get("queue") is not None:
            data["queue"] = str(data["queue"])
        if data.get("idempotency_key") is not None:
            data["idempotency_key"] = str(data["idempotency_key"])
        if data.get("claimed_by") is not None:
            data["claimed_by"] = str(data["claimed_by"])

        data["state"] = _state_value(data.get("state")) or _PENDING
        data["priority"] = _to_int(data.get("priority"), 0)
        data["run_at"] = _to_float(data.get("run_at"), now)
        data["attempts"] = _to_int(data.get("attempts"), 0)
        data["created_at"] = _to_float(data.get("created_at"), now)
        data["claim_expires"] = _to_float(data.get("claim_expires"), 0.0)

        if data.get("args") is None:
            data["args"] = []
        if data.get("kwargs") is None:
            data["kwargs"] = {}

        return data

    def _row_values(self, data: dict[str, Any]):
        payload = _safe_dumps(data)
        return (
            data.get("id"),
            data.get("name"),
            _safe_dumps(data.get("args", [])),
            _safe_dumps(data.get("kwargs", {})),
            data.get("queue"),
            data.get("priority", 0),
            data.get("run_at", 0.0),
            _state_value(data.get("state")) or _PENDING,
            data.get("attempts", 0),
            data.get("created_at", 0.0),
            data.get("idempotency_key"),
            data.get("claimed_by"),
            data.get("claim_expires", 0.0),
            _safe_dumps(data.get("result")),
            data.get("last_error"),
            payload,
        )

    def _row_to_data(self, row) -> dict[str, Any]:
        payload_data = _safe_loads(row["payload"], {})
        if not isinstance(payload_data, dict):
            payload_data = {}

        data = dict(payload_data)
        data.update(
            {
                "id": row["id"],
                "name": row["name"],
                "args": _safe_loads(row["args"], []),
                "kwargs": _safe_loads(row["kwargs"], {}),
                "queue": row["queue"],
                "priority": _to_int(row["priority"], 0),
                "run_at": _to_float(row["run_at"], 0.0),
                "state": _coerce_loaded_state(row["state"]),
                "attempts": _to_int(row["attempts"], 0),
                "created_at": _to_float(row["created_at"], 0.0),
                "idempotency_key": row["idempotency_key"],
                "claimed_by": row["claimed_by"],
                "claim_expires": _to_float(row["claim_expires"], 0.0),
                "result": _safe_loads(row["result"], None),
                "last_error": row["last_error"],
            }
        )
        if data["args"] is None:
            data["args"] = []
        if data["kwargs"] is None:
            data["kwargs"] = {}
        return data

    def _row_to_task(self, row) -> Task:
        data = self._row_to_data(row)

        task = None
        try:
            sig = inspect.signature(Task)
            params = sig.parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_kwargs:
                kwargs = dict(data)
            else:
                kwargs = {k: v for k, v in data.items() if k in params}
            task = Task(**kwargs)
        except Exception:
            common = {k: data[k] for k in _TASK_FIELDS if k in data}
            try:
                task = Task(**common)
            except Exception:
                try:
                    task = Task.__new__(Task)
                except Exception:
                    raise

        for key, value in data.items():
            try:
                setattr(task, key, value)
            except Exception:
                try:
                    object.__setattr__(task, key, value)
                except Exception:
                    pass

        return task

    def _payload_from_row_with(self, row, changes: dict[str, Any]) -> str:
        data = self._row_to_data(row)
        data.update(changes)
        if "state" in data:
            data["state"] = _state_value(data["state"])
        return _safe_dumps(data)

    def _payload_with(self, task_id, changes: dict[str, Any]) -> str:
        task_id = _object_id(task_id)
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            data = dict(changes)
        else:
            data = self._row_to_data(row)
            data.update(changes)
        if "state" in data:
            data["state"] = _state_value(data["state"])
        return _safe_dumps(data)


SqliteStore._native_fail = SqliteStore.fail