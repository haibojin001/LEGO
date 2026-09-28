from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from ..settings import PATHS
from .registry import Tool, ToolRegistry


def _safe_name(name: str) -> str:
    allowed = set("-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    result = "".join(character if character in allowed else "_" for character in name)
    result = result.strip("_")
    if not result:
        result = "script"
    return result[:64]


def _write_and_run(
    *,
    language: str,
    source: str,
    name: str,
    args: list[str] | None = None,
    timeout: int = 180,
) -> dict:
    PATHS.ensure()

    extensions = {
        "python": ".py",
        "bash": ".sh",
        "node": ".js",
    }
    filename = f"{int(time.time())}-{_safe_name(name)}{extensions.get(language, '.txt')}"
    script_file = PATHS.scripts / filename
    script_file.write_text(source)

    if language == "bash":
        current_mode = os.stat(script_file).st_mode
        os.chmod(script_file, current_mode | stat.S_IEXEC | stat.S_IXGRP)

    supplied_args = args or []
    invocations = {
        "python": [sys.executable, str(script_file), *supplied_args],
        "bash": ["bash", str(script_file), *supplied_args],
        "node": ["node", str(script_file), *supplied_args],
    }
    invocation = invocations.get(language)

    if invocation is None:
        return {"error": f"unsupported language: {language}"}

    try:
        process = subprocess.run(
            invocation,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "path": str(script_file),
            "exit_code": -1,
            "stdout": "",
            "stderr": f"timed out after {timeout}s",
        }

    return {
        "path": str(script_file),
        "exit_code": process.returncode,
        "stdout": process.stdout[-8000:],
        "stderr": process.stderr[-4000:],
    }


def register(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="script.run",
            description=(
                "Persist source code to ~/.automate/scripts/ then run it. "
                "Use this when a one-shot multi-line script is cleaner than chaining shell.exec calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "bash", "node"],
                    },
                    "source": {
                        "type": "string",
                        "description": "Full script source.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short label, used in the saved filename.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 180,
                    },
                },
                "required": ["language", "source", "name"],
            },
            handler=_write_and_run,
            category="system",
            danger="high",
        )
    )

    def _list_scripts() -> dict:
        found = []
        ordered = sorted(
            PATHS.scripts.glob("*"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        for candidate in ordered[:50]:
            found.append(
                {
                    "name": candidate.name,
                    "size": candidate.stat().st_size,
                    "mtime": candidate.stat().st_mtime,
                }
            )
        return {"scripts": found}

    reg.register(
        Tool(
            name="script.list",
            description="List saved scripts (most recent first, capped at 50).",
            parameters={"type": "object", "properties": {}},
            handler=_list_scripts,
            category="system",
        )
    )

    def _read_script(name: str) -> dict:
        candidate = PATHS.scripts / _safe_name(name)
        if not candidate.exists():
            candidate = PATHS.scripts / name
        if not candidate.exists() or not candidate.is_file():
            return {"error": "not found"}
        return {"name": candidate.name, "source": candidate.read_text()}

    reg.register(
        Tool(
            name="script.read",
            description="Read a previously saved script's source by filename.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=_read_script,
            category="system",
        )
    )