from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from ..store import get_db
from .registry import Tool, ToolRegistry


def _to_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if "T" not in value and " " not in value:
            value = f"{value}T00:00"
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _fts_query(raw: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in raw
    )
    return " ".join(
        f"{part}*"
        for part in cleaned.split()
        if len(part) >= 2
    )


def find(
    *,
    query: str = "",
    kinds: list[str] | None = None,
    tags: str = "",
    date_after: str = "",
    date_before: str = "",
    limit: int = 20,
) -> dict:
    db = get_db()
    requested_kinds = set(kinds or ["notes", "files"])
    tag_pattern = f"%,{tags.strip()},%" if tags.strip() else None
    after = _to_ts(date_after)
    before = _to_ts(date_before)
    fts = _fts_query(query) if query else ""

    results: list[dict] = []

    if "notes" in requested_kinds:
        results.extend(_search_notes(db, fts, tag_pattern, after, before, limit))
    if "files" in requested_kinds:
        results.extend(_search_files(db, fts, tag_pattern, after, before, limit))

    results.sort(key=lambda row: row.get("rank", 0))
    return {
        "results": results[:limit],
        "query": query,
        "matched": len(results),
    }


def _search_notes(
    db: Any,
    fts: str,
    tag_pat: str | None,
    after: float | None,
    before: float | None,
    limit: int,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []

    if fts:
        clauses.append(
            "notes.rowid IN (SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?)"
        )
        params.append(fts)
    if tag_pat:
        clauses.append("(',' || notes.tags || ',') LIKE ?")
        params.append(tag_pat)
    if after is not None:
        clauses.append("notes.updated_at >= ?")
        params.append(after)
    if before is not None:
        clauses.append("notes.updated_at < ?")
        params.append(before)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rank = "bm25(notes_fts)" if fts else "0"
    join = "LEFT JOIN notes_fts ON notes_fts.rowid = notes.rowid " if fts else ""

    sql = (
        "SELECT notes.id AS id, notes.title AS title, "
        "substr(notes.body, 1, 200) AS snippet, "
        "notes.tags AS tags, notes.updated_at AS updated_at, "
        f"{rank} AS rank, 'note' AS kind "
        "FROM notes "
        f"{join}"
        f"{where}"
        " ORDER BY rank, notes.updated_at DESC LIMIT ?"
    )
    params.append(limit)

    try:
        return db.fetchall(sql, params)
    except sqlite3.OperationalError:
        return []


def _search_files(
    db: Any,
    fts: str,
    tag_pat: str | None,
    after: float | None,
    before: float | None,
    limit: int,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []

    if fts:
        clauses.append(
            "files_meta.rowid IN (SELECT rowid FROM files_fts WHERE files_fts MATCH ?)"
        )
        params.append(fts)
    if tag_pat:
        clauses.append("(',' || files_meta.tags || ',') LIKE ?")
        params.append(tag_pat)
    if after is not None:
        clauses.append("files_meta.created_at >= ?")
        params.append(after)
    if before is not None:
        clauses.append("files_meta.created_at < ?")
        params.append(before)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rank = "bm25(files_fts)" if fts else "0"
    join = (
        "LEFT JOIN files_fts ON files_fts.rowid = files_meta.rowid "
        if fts
        else ""
    )

    sql = (
        "SELECT files_meta.id AS id, files_meta.filename AS title, "
        "files_meta.description AS snippet, files_meta.tags AS tags, "
        "files_meta.created_at AS updated_at, "
        f"{rank} AS rank, 'file' AS kind "
        "FROM files_meta "
        f"{join}"
        f"{where}"
        " ORDER BY rank, files_meta.created_at DESC LIMIT ?"
    )
    params.append(limit)

    try:
        return db.fetchall(sql, params)
    except sqlite3.OperationalError:
        return []


def register(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="search.find",
            description=(
                "Find a specific note or file the user has stored. Use this "
                "(not notes.search / files.list) whenever the user asks "
                "'find / search / where is / show me X' — pass the parsed "
                "criteria, do not list everything and grep manually. Returns "
                "BM25-ranked results across notes + files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text terms — proper nouns and IDs work well.",
                    },
                    "kinds": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["notes", "files"],
                        },
                        "description": "Restrict to notes / files; default both.",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Single tag to filter by.",
                    },
                    "date_after": {
                        "type": "string",
                        "description": "ISO date — only items updated/created on or after this.",
                    },
                    "date_before": {
                        "type": "string",
                        "description": "ISO date — only items before this.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max results across both kinds.",
                    },
                },
            },
            handler=find,
            category="search",
        )
    )