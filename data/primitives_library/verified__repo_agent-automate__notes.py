from __future__ import annotations

import time
import uuid
from typing import Any

from .store import Database


def _norm_tags(raw: str) -> str:
    tags = {
        part.strip()
        for part in (raw or "").split(",")
        if part.strip()
    }
    return ",".join(sorted(tags))


def _fts_query(raw: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in raw
    )
    return " ".join(
        f"{word}*"
        for word in cleaned.split()
        if len(word) >= 2
    )


def get_note(db: Database, note_id: str) -> dict | None:
    return db.fetchone(
        "SELECT * FROM notes WHERE id = ?",
        (note_id,),
    )


def create_note(
    db: Database,
    *,
    title: str,
    body: str = "",
    tags: str = "",
    pinned: bool = False,
) -> dict:
    note_id = uuid.uuid4().hex
    now = time.time()

    db.execute(
        (
            "INSERT INTO notes "
            "(id, title, body, tags, pinned, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            note_id,
            title.strip() or "Untitled",
            body,
            _norm_tags(tags),
            1 if pinned else 0,
            now,
            now,
        ),
    )
    return get_note(db, note_id)  # type: ignore[return-value]


def update_note(
    db: Database,
    note_id: str,
    *,
    title: str | None = None,
    body: str | None = None,
    tags: str | None = None,
    pinned: bool | None = None,
) -> dict | None:
    current = get_note(db, note_id)
    if not current:
        return None

    next_title = current["title"] if title is None else title
    next_body = current["body"] if body is None else body
    next_tags = current["tags"] if tags is None else _norm_tags(tags)
    next_pinned: Any = (
        current["pinned"]
        if pinned is None
        else (1 if pinned else 0)
    )

    db.execute(
        (
            "UPDATE notes "
            "SET title=?, body=?, tags=?, pinned=?, updated_at=? "
            "WHERE id=?"
        ),
        (
            next_title,
            next_body,
            next_tags,
            next_pinned,
            time.time(),
            note_id,
        ),
    )
    return get_note(db, note_id)


def list_notes(
    db: Database,
    *,
    query: str = "",
    tag: str = "",
    limit: int = 200,
) -> list[dict]:
    conditions: list[str] = ["1=1"]
    values: list[Any] = []

    if query:
        match = _fts_query(query)
        if match:
            try:
                db.fetchone("SELECT 1 FROM notes_fts LIMIT 1")
            except Exception:
                pattern = f"%{query}%"
                conditions.append("(title LIKE ? OR body LIKE ?)")
                values.extend((pattern, pattern))
            else:
                conditions.append(
                    "rowid IN ("
                    "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?"
                    ")"
                )
                values.append(match)
        else:
            pattern = f"%{query}%"
            conditions.append("(title LIKE ? OR body LIKE ?)")
            values.extend((pattern, pattern))

    if tag:
        conditions.append("(',' || tags || ',') LIKE ?")
        values.append(f"%,{tag.strip()},%")

    statement = (
        f"SELECT * FROM notes WHERE {' AND '.join(conditions)} "
        "ORDER BY pinned DESC, updated_at DESC LIMIT ?"
    )
    values.append(limit)
    return db.fetchall(statement, tuple(values))


def delete_note(db: Database, note_id: str) -> bool:
    if not get_note(db, note_id):
        return False
    db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return True


def all_tags(db: Database) -> list[str]:
    rows = db.fetchall(
        "SELECT DISTINCT tags FROM notes WHERE tags != ''"
    )
    found: set[str] = set()

    for row in rows:
        for tag in row["tags"].split(","):
            cleaned = tag.strip()
            if cleaned:
                found.add(cleaned)

    return sorted(found)