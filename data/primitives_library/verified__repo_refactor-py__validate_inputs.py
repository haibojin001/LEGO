from __future__ import annotations

from argparse import Namespace
from pathlib import Path

_DEFAULT_FILES = [Path("refactors.py"), Path(".refactors/__init__.py")]


def validate_main_inputs(options: Namespace) -> None:
    refactor_file = options.refactor_file

    if refactor_file:
        if not refactor_file.exists():
            raise ValueError(
                f"Given --refactor-file '{refactor_file!s}' doesn't exist"
            )
        return

    for default_file in _DEFAULT_FILES:
        if default_file.exists():
            options.refactor_file = default_file
            return

    raise ValueError(
        "Either provide a file using --refactor-file or ensure one of "
        "these directories exist: "
        + ", ".join(str(path) for path in _DEFAULT_FILES)
    )