from __future__ import annotations

import os
import subprocess
from typing import Any

from s_tui.sources.hook import Hook


class ScriptHook:
    """
    Runs an arbitrary shell script stored in the filesystem when invoked.
    """

    def __init__(self, path: str, timeout_milliseconds: int = 0) -> None:
        self.path = path
        self.hook = self._make_script_hook(path, timeout_milliseconds)

    def is_ready(self) -> bool:
        return self.hook.is_ready()

    def invoke(self) -> None:
        self.hook.invoke()

    def _run_script(self, *args: Any) -> None:
        with open(os.devnull, "w") as dev_null:
            subprocess.Popen(
                ["/bin/sh", args[0][0]],
                stdout=dev_null,
                stderr=dev_null,
            )

    def _make_script_hook(self, path: str, timeout_milliseconds: int) -> Hook:
        return Hook(self._run_script, timeout_milliseconds, path)