from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from . import push
from .store import Database

log = logging.getLogger("automate.reminders")

_RECUR_DELTA = {
    "minutely": timedelta(minutes=1),
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
}


def create(
    db: Database,
    *,
    body: str,
    due_at: float,
    recurrence: str | None = None,
    context: dict | None = None,
) -> dict:
    reminder_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO reminders "
        "(id, body, due_at, recurrence, status, context_json, created_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
        (
            reminder_id,
            body,
            float(due_at),
            recurrence,
            json.dumps(context or {}),
            time.time(),
        ),
    )
    return get(db, reminder_id)  # type: ignore[return-value]


def get(db: Database, reminder_id: str) -> dict | None:
    return db.fetchone(
        "SELECT * FROM reminders WHERE id = ?",
        (reminder_id,),
    )


def list_all(
    db: Database,
    *,
    status: str | None = None,
    limit: int = 200,
) -> list[dict]:
    if status:
        return db.fetchall(
            "SELECT * FROM reminders WHERE status = ? ORDER BY due_at ASC LIMIT ?",
            (status, limit),
        )
    return db.fetchall(
        "SELECT * FROM reminders ORDER BY due_at ASC LIMIT ?",
        (limit,),
    )


def snooze(
    db: Database,
    reminder_id: str,
    *,
    minutes: int,
) -> dict | None:
    reminder = get(db, reminder_id)
    if not reminder:
        return None

    due_at = max(time.time(), reminder["due_at"]) + minutes * 60
    db.execute(
        "UPDATE reminders SET due_at = ?, status = 'pending', fired_at = NULL WHERE id = ?",
        (due_at, reminder_id),
    )
    return get(db, reminder_id)


def dismiss(db: Database, reminder_id: str) -> bool:
    if not get(db, reminder_id):
        return False
    db.execute(
        "UPDATE reminders SET status = 'dismissed' WHERE id = ?",
        (reminder_id,),
    )
    return True


def delete(db: Database, reminder_id: str) -> bool:
    if not get(db, reminder_id):
        return False
    db.execute(
        "DELETE FROM reminders WHERE id = ?",
        (reminder_id,),
    )
    return True


class Scheduler:
    """Background worker responsible for dispatching due reminders."""

    def __init__(self, db: Database, *, poll_seconds: float = 30.0):
        self.db = db
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="automate-reminders",
        )
        self._thread.start()
        log.info("reminder scheduler started")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception as exc:
                log.exception("scheduler tick failed: %s", exc)

    def tick(self) -> int:
        now = time.time()
        reminders = self.db.fetchall(
            "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ?",
            (now,),
        )
        for reminder in reminders:
            self._fire(reminder, now)
        return len(reminders)

    def _fire(self, reminder: dict, now: float) -> None:
        log.info(
            "firing reminder %s: %s",
            reminder["id"],
            reminder["body"][:60],
        )
        self.db.execute(
            "UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ?",
            (now, reminder["id"]),
        )
        try:
            push.push(
                self.db,
                title="autoMate reminder",
                body=reminder["body"],
                url=f"/?reminder={reminder['id']}",
                tag=f"reminder-{reminder['id']}",
            )
        except Exception as exc:
            log.warning("push fanout failed for %s: %s", reminder["id"], exc)

        recurrence = reminder.get("recurrence")
        interval = _RECUR_DELTA.get(recurrence or "")
        if interval:
            next_due_at = (
                datetime.fromtimestamp(reminder["due_at"]) + interval
            ).timestamp()
            self.db.execute(
                "INSERT INTO reminders "
                "(id, body, due_at, recurrence, status, context_json, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (
                    uuid.uuid4().hex,
                    reminder["body"],
                    next_due_at,
                    recurrence,
                    reminder.get("context_json") or "{}",
                    time.time(),
                ),
            )