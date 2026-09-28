from __future__ import annotations

import hashlib
import mimetypes
import time
import uuid
from pathlib import Path

from .settings import PATHS
from .store import Database

STORAGE_DIR_KEY = "files.storage_dir"


def storage_dir(db: Database) -> Path:
    configured = db.get_setting(STORAGE_DIR_KEY)
    if configured:
        try:
            directory = Path(configured).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        except OSError:
            pass
    PATHS.ensure()
    return PATHS.files


def set_storage_dir(db: Database, path: str) -> Path:
    if not path or not path.strip():
        raise ValueError("storage path is empty")

    directory = Path(path.strip()).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"can't create directory: {exc}") from exc

    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")

    probe = directory / ".automate_write_probe"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise ValueError(f"directory is not writable: {exc}") from exc

    db.set_setting(STORAGE_DIR_KEY, str(directory))
    return directory


def store_blob(
    content: bytes,
    *,
    filename: str,
    mime: str | None = None,
    tags: str = "",
    description: str = "",
    db: Database,
) -> dict:
    directory = storage_dir(db)
    digest = hashlib.sha256(content).hexdigest()
    destination = directory / digest

    if not destination.exists():
        destination.write_bytes(content)

    file_id = uuid.uuid4().hex
    detected_mime = mime or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    db.execute(
        (
            "INSERT INTO files_meta "
            "(id, sha256, filename, mime, size, tags, description, created_at, storage_root) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            file_id,
            digest,
            filename,
            detected_mime,
            len(content),
            _norm_tags(tags),
            description,
            time.time(),
            str(directory),
        ),
    )
    return get_meta(db, file_id)  # type: ignore[return-value]


def get_meta(db: Database, file_id: str) -> dict | None:
    return db.fetchone("SELECT * FROM files_meta WHERE id = ?", (file_id,))


def open_blob(meta: dict) -> Path:
    root = Path(meta["storage_root"]) if meta.get("storage_root") else PATHS.files
    return root / meta["sha256"]


def list_files(
    db: Database,
    *,
    query: str = "",
    tag: str = "",
    limit: int = 200,
) -> list[dict]:
    conditions: list[str] = ["1=1"]
    values: list[object] = []

    if query:
        match_query = _fts_query(query)
        if match_query:
            try:
                db.fetchone("SELECT 1 FROM files_fts LIMIT 1")
                conditions.append(
                    "rowid IN (SELECT rowid FROM files_fts WHERE files_fts MATCH ?)"
                )
                values.append(match_query)
            except Exception:
                conditions.append("(filename LIKE ? OR description LIKE ?)")
                pattern = f"%{query}%"
                values.extend((pattern, pattern))
        else:
            conditions.append("(filename LIKE ? OR description LIKE ?)")
            pattern = f"%{query}%"
            values.extend((pattern, pattern))

    if tag:
        conditions.append("(',' || tags || ',') LIKE ?")
        values.append(f"%,{tag.strip()},%")

    statement = (
        f"SELECT * FROM files_meta WHERE {' AND '.join(conditions)} "
        "ORDER BY created_at DESC LIMIT ?"
    )
    values.append(limit)
    return db.fetchall(statement, tuple(values))


def _fts_query(raw: str) -> str:
    safe = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in raw
    )
    return " ".join(
        f"{part}*" for part in safe.split() if len(part) >= 2
    )


def delete_file(db: Database, file_id: str) -> bool:
    meta = get_meta(db, file_id)
    if not meta:
        return False

    db.execute("DELETE FROM files_meta WHERE id = ?", (file_id,))
    remaining = db.fetchone(
        "SELECT 1 FROM files_meta WHERE sha256 = ? LIMIT 1",
        (meta["sha256"],),
    )
    if not remaining:
        try:
            open_blob(meta).unlink()
        except FileNotFoundError:
            pass
    return True


def total_size(db: Database) -> int:
    result = db.fetchone(
        "SELECT COALESCE(SUM(size), 0) AS total FROM files_meta"
    )
    return int(result["total"]) if result else 0


def _norm_tags(raw: str) -> str:
    return ",".join(
        sorted(
            {
                value.strip()
                for value in (raw or "").split(",")
                if value.strip()
            }
        )
    )