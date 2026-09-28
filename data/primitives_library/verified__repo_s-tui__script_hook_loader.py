from __future__ import annotations

import os

from s_tui.sources.hook_script import ScriptHook


class ScriptHookLoader:
    """
    Creates script hooks from shell scripts found in a hooks.d directory.
    """

    def __init__(self, dir_path: str) -> None:
        self.scripts_dir_path = os.path.join(dir_path, "hooks.d")

    def load_script(
        self, source_name: str, timeoutMilliseconds: int = 0
    ) -> ScriptHook | None:
        """
        Return a script hook for the named source, if its script exists.
        """
        script_path = os.path.join(
            self.scripts_dir_path,
            self._source_to_script_name(source_name),
        )
        if not os.path.isfile(script_path):
            return None
        return ScriptHook(script_path, timeoutMilliseconds)

    def _source_to_script_name(self, source_name: str) -> str:
        return f"{source_name.lower()}.sh"