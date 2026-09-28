from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from ..settings import PATHS
from .crypto import Vault


_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    base_url TEXT,
    api_key_enc TEXT,
    default_model TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    extra_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS connections (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    auth_kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'disconnected',
    token_enc TEXT,
    refresh_enc TEXT,
    expires_at REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    trace_json TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    started_at REAL NOT NULL,
    ended_at REAL
);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_updated_idx ON notes(updated_at DESC);

CREATE TABLE IF NOT EXISTS files_meta (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL DEFAULT 'application/octet-stream',
    size INTEGER NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    storage_root TEXT
);
CREATE INDEX IF NOT EXISTS files_sha_idx ON files_meta(sha256);
CREATE INDEX IF NOT EXISTS files_created_idx ON files_meta(created_at DESC);

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    due_at REAL NOT NULL,
    recurrence TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    fired_at REAL,
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS reminders_due_idx ON reminders(status, due_at);

CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint TEXT PRIMARY KEY,
    p256dh TEXT NOT NULL,
    auth TEXT NOT NULL,
    user_agent TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS push_keys (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    vapid_private TEXT NOT NULL,
    vapid_public TEXT NOT NULL,
    vapid_subject TEXT NOT NULL DEFAULT 'mailto:admin@automate.local'
);

CREATE TABLE IF NOT EXISTS bots (
    id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    config_enc TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'stopped',
    last_error TEXT,
    updated_at REAL NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, body, tags,
    content='notes', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, body, tags)
    VALUES (new.rowid, new.title, new.body, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, body, tags)
    VALUES ('delete', old.rowid, old.title, old.body, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, body, tags)
    VALUES ('delete', old.rowid, old.title, old.body, old.tags);
    INSERT INTO notes_fts(rowid, title, body, tags)
    VALUES (new.rowid, new.title, new.body, new.tags);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    filename, description, tags,
    content='files_meta', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files_meta BEGIN
    INSERT INTO files_fts(rowid, filename, description, tags)
    VALUES (new.rowid, new.filename, new.description, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files_meta BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, description, tags)
    VALUES ('delete', old.rowid, old.filename, old.description, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files_meta BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, description, tags)
    VALUES ('delete', old.rowid, old.filename, old.description, old.tags);
    INSERT INTO files_fts(rowid, filename, description, tags)
    VALUES (new.rowid, new.filename, new.description, new.tags);
END;
"""


class Database:
    def __init__(self, path: Path | None = None, vault: Vault | None = None):
        self.path = Path(path or PATHS.db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate_v452()
        try:
            self._conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            pass
        self._backfill_fts()
        self.vault = vault or Vault(PATHS.secret_key)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate_v452(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(files_meta)")
        }
        if "storage_root" not in columns:
            self._conn.execute("ALTER TABLE files_meta ADD COLUMN storage_root TEXT")

    def _backfill_fts(self) -> None:
        try:
            for table, index, columns in (
                ("notes", "notes_fts", "title, body, tags"),
                ("files_meta", "files_fts", "filename, description, tags"),
            ):
                count = self._conn.execute(
                    f"SELECT COUNT(*) FROM {index}"
                ).fetchone()[0]
                if not count:
                    self._conn.execute(
                        f"INSERT INTO {index}(rowid, {columns}) "
                        f"SELECT rowid, {columns} FROM {table}"
                    )
        except sqlite3.OperationalError:
            pass

    @contextmanager
    def cursor(self):
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self.cursor() as cursor:
            cursor.execute(sql, tuple(params))

    @staticmethod
    def _json(value: Any, fallback: str = "{}") -> str:
        if value is None:
            return fallback
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _decoded(value: Any, fallback: Any) -> Any:
        if not isinstance(value, str):
            return fallback
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return fallback

    def _encrypt(self, value: str | None) -> str | None:
        return None if value is None else self.vault.encrypt(value)

    def _decrypt(self, value: str | None) -> str | None:
        if not value:
            return None
        try:
            return self.vault.decrypt(value)
        except Exception:
            return None

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value FROM settings WHERE key = ?", (key,))
        return default if row is None else row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        self.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, text),
        )

    def delete_setting(self, key: str) -> None:
        self.execute("DELETE FROM settings WHERE key = ?", (key,))

    def list_providers(self) -> list[dict[str, Any]]:
        rows = self.fetchall("SELECT * FROM providers ORDER BY updated_at DESC")
        return [self._provider_row(row) for row in rows]

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM providers WHERE id = ?", (provider_id,))
        return self._provider_row(row) if row else None

    def _provider_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["enabled"] = bool(row["enabled"])
        row["extra"] = self._decoded(row.pop("extra_json", "{}"), {})
        row["api_key"] = self._decrypt(row.get("api_key_enc"))
        return row

    def save_provider(
        self,
        provider_id: str,
        display_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        enabled: bool = True,
        extra: Any = None,
        **kwargs: Any,
    ) -> None:
        existing = self.get_provider(provider_id)
        if api_key is None and existing is not None:
            api_key_enc = existing.get("api_key_enc")
        else:
            api_key_enc = self._encrypt(api_key)
        if extra is None:
            extra = kwargs.get("extra_json", existing.get("extra", {}) if existing else {})
        now = time.time()
        self.execute(
            "INSERT INTO providers(id, display_name, base_url, api_key_enc, default_model, "
            "enabled, extra_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name, "
            "base_url=excluded.base_url, api_key_enc=excluded.api_key_enc, "
            "default_model=excluded.default_model, enabled=excluded.enabled, "
            "extra_json=excluded.extra_json, updated_at=excluded.updated_at",
            (
                provider_id,
                display_name,
                base_url,
                api_key_enc,
                default_model,
                int(enabled),
                self._json(extra),
                now,
            ),
        )

    upsert_provider = save_provider

    def delete_provider(self, provider_id: str) -> None:
        self.execute("DELETE FROM providers WHERE id = ?", (provider_id,))

    def list_connections(self) -> list[dict[str, Any]]:
        return [self._connection_row(row) for row in self.fetchall(
            "SELECT * FROM connections ORDER BY updated_at DESC"
        )]

    def get_connection(self, connection_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM connections WHERE id = ?", (connection_id,))
        return self._connection_row(row) if row else None

    def _connection_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["metadata"] = self._decoded(row.pop("metadata_json", "{}"), {})
        row["token"] = self._decrypt(row.get("token_enc"))
        row["refresh_token"] = self._decrypt(row.get("refresh_enc"))
        return row

    def save_connection(
        self,
        connection_id: str,
        display_name: str,
        auth_kind: str,
        status: str = "disconnected",
        token: str | None = None,
        refresh_token: str | None = None,
        expires_at: float | None = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        old = self.get_connection(connection_id)
        token_enc = old.get("token_enc") if token is None and old else self._encrypt(token)
        refresh_enc = (
            old.get("refresh_enc")
            if refresh_token is None and old
            else self._encrypt(refresh_token)
        )
        if metadata is None:
            metadata = kwargs.get("metadata_json", old.get("metadata", {}) if old else {})
        self.execute(
            "INSERT INTO connections(id, display_name, auth_kind, status, token_enc, "
            "refresh_enc, expires_at, metadata_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name, "
            "auth_kind=excluded.auth_kind, status=excluded.status, token_enc=excluded.token_enc, "
            "refresh_enc=excluded.refresh_enc, expires_at=excluded.expires_at, "
            "metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
            (
                connection_id, display_name, auth_kind, status, token_enc, refresh_enc,
                expires_at, self._json(metadata), time.time(),
            ),
        )

    upsert_connection = save_connection

    def delete_connection(self, connection_id: str) -> None:
        self.execute("DELETE FROM connections WHERE id = ?", (connection_id,))

    def create_run(
        self,
        run_id: str,
        source: str,
        prompt: str,
        status: str = "running",
        trace: Any = None,
        result: str | None = None,
        started_at: float | None = None,
    ) -> None:
        self.execute(
            "INSERT INTO runs(id, source, prompt, status, trace_json, result, started_at, ended_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (run_id, source, prompt, status, self._json(trace, "[]"), result, started_at or time.time()),
        )

    def update_run(self, run_id: str, **values: Any) -> None:
        allowed = {"source", "prompt", "status", "result", "started_at", "ended_at"}
        fields: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            if key in {"trace", "trace_json"}:
                fields.append("trace_json = ?")
                params.append(self._json(value, "[]"))
            elif key in allowed:
                fields.append(f"{key} = ?")
                params.append(value)
        if fields:
            params.append(run_id)
            self.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id = ?", params)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
        return self._run_row(row) if row else None

    def list_runs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return [
            self._run_row(row)
            for row in self.fetchall(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        ]

    def _run_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["trace"] = self._decoded(row.pop("trace_json", "[]"), [])
        return row

    def save_note(self, note_id: str, title: str, body: str = "", tags: str = "",
                  pinned: bool = False, created_at: float | None = None) -> None:
        now = time.time()
        self.execute(
            "INSERT INTO notes(id, title, body, tags, pinned, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "title=excluded.title, body=excluded.body, tags=excluded.tags, "
            "pinned=excluded.pinned, updated_at=excluded.updated_at",
            (note_id, title, body, tags, int(pinned), created_at or now, now),
        )

    create_note = save_note

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM notes WHERE id = ?", (note_id,))
        if row:
            row["pinned"] = bool(row["pinned"])
        return row

    def list_notes(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM notes ORDER BY pinned DESC, updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        for row in rows:
            row["pinned"] = bool(row["pinned"])
        return rows

    def update_note(self, note_id: str, **values: Any) -> None:
        allowed = {"title", "body", "tags", "pinned"}
        fields, params = [], []
        for key, value in values.items():
            if key in allowed and value is not None:
                fields.append(f"{key} = ?")
                params.append(int(value) if key == "pinned" else value)
        if fields:
            fields.append("updated_at = ?")
            params.extend((time.time(), note_id))
            self.execute(f"UPDATE notes SET {', '.join(fields)} WHERE id = ?", params)

    def delete_note(self, note_id: str) -> None:
        self.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def save_file_meta(self, file_id: str, sha256: str, filename: str, size: int,
                       mime: str = "application/octet-stream", tags: str = "",
                       description: str = "", storage_root: str | None = None,
                       created_at: float | None = None) -> None:
        self.execute(
            "INSERT INTO files_meta(id, sha256, filename, mime, size, tags, description, created_at, storage_root) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "sha256=excluded.sha256, filename=excluded.filename, mime=excluded.mime, "
            "size=excluded.size, tags=excluded.tags, description=excluded.description, "
            "storage_root=excluded.storage_root",
            (file_id, sha256, filename, mime, size, tags, description, created_at or time.time(), storage_root),
        )

    def get_file_meta(self, file_id: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM files_meta WHERE id = ?", (file_id,))

    def list_files(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self.fetchall(
            "SELECT * FROM files_meta ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    def delete_file_meta(self, file_id: str) -> None:
        self.execute("DELETE FROM files_meta WHERE id = ?", (file_id,))

    def get_memory(self, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value FROM memory WHERE key = ?", (key,))
        return default if row is None else row["value"]

    def set_memory(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO memory(key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), time.time()),
        )

    def delete_memory(self, key: str) -> None:
        self.execute("DELETE FROM memory WHERE key = ?", (key,))

    def list_memory(self) -> list[dict[str, Any]]:
        return self.fetchall("SELECT * FROM memory ORDER BY updated_at DESC")


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database()
    return _db