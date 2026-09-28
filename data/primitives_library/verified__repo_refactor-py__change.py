from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from refactor.ast import split_lines
from refactor.common import _FileInfo


@dataclass
class Change:
    file_info: _FileInfo
    original_source: str
    refactored_source: str

    def __post_init__(self) -> None:
        if self.file_info.path is None:
            raise ValueError("Can't apply a change to a string")

    @property
    def file(self) -> Path:
        path = self.file_info.path
        if path is None:
            raise ValueError("Change expects a valid file")
        return path

    def compute_diff(self) -> str:
        before = split_lines(self.original_source)
        after = split_lines(self.refactored_source)
        return "".join(
            difflib.unified_diff(
                before,
                after,
                os.fspath(self.file),
                os.fspath(self.file),
            )
        )

    def apply_diff(self) -> None:
        data = self.refactored_source.encode(self.file_info.get_encoding())
        with open(self.file, "wb") as stream:
            stream.write(data)