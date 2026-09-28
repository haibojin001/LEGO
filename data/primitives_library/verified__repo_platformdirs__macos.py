from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from ._xdg import XDGMixin
from .api import PlatformDirsABC

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _MacOSDefaults(PlatformDirsABC):
    def _with_app_parts(self, root: str) -> str:
        return self._append_app_name_and_version(root)

    def _homebrew_prefix(self) -> str | None:
        marker = "/opt/python"
        if marker not in sys.prefix:
            return None
        return sys.prefix.split(marker)[0]

    def _shared_support_dirs(self) -> list[str]:
        prefix = self._homebrew_prefix()
        result: list[str] = []
        if prefix is not None:
            result.append(self._with_app_parts(f"{prefix}/share"))
        result.append(self._with_app_parts("/Library/Application Support"))
        return result

    @property
    def user_data_dir(self) -> str:
        return self._with_app_parts(os.path.expanduser("~/Library/Application Support"))

    @property
    def _site_data_dirs(self) -> list[str]:
        return self._shared_support_dirs()

    @property
    def site_data_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_data_dir)

    @property
    def user_config_dir(self) -> str:
        return self.user_data_dir

    @property
    def _site_config_dirs(self) -> list[str]:
        return self._shared_support_dirs()

    @property
    def site_config_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_config_dir)

    @property
    def user_cache_dir(self) -> str:
        return self._with_app_parts(os.path.expanduser("~/Library/Caches"))

    @property
    def _site_cache_dirs(self) -> list[str]:
        prefix = self._homebrew_prefix()
        result: list[str] = []
        if prefix is not None:
            result.append(self._with_app_parts(f"{prefix}/var/cache"))
        result.append(self._with_app_parts("/Library/Caches"))
        return result

    @property
    def site_cache_dir(self) -> str:
        directories = self._site_cache_dirs
        if self.multipath:
            return os.pathsep.join(directories)
        return directories[0]

    @property
    def site_cache_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_cache_dir)

    @property
    def user_state_dir(self) -> str:
        return self.user_data_dir

    @property
    def site_state_dir(self) -> str:
        return self._shared_support_dirs()[0]

    @property
    def user_log_dir(self) -> str:
        return self._with_app_parts(os.path.expanduser("~/Library/Logs"))

    @property
    def site_log_dir(self) -> str:
        return self._with_app_parts("/Library/Logs")

    @property
    def user_documents_dir(self) -> str:
        return os.path.expanduser("~/Documents")

    @property
    def user_downloads_dir(self) -> str:
        return os.path.expanduser("~/Downloads")

    @property
    def user_pictures_dir(self) -> str:
        return os.path.expanduser("~/Pictures")

    @property
    def user_videos_dir(self) -> str:
        return os.path.expanduser("~/Movies")

    @property
    def user_music_dir(self) -> str:
        return os.path.expanduser("~/Music")

    @property
    def user_desktop_dir(self) -> str:
        return os.path.expanduser("~/Desktop")

    @property
    def user_projects_dir(self) -> str:
        return os.path.expanduser("~/Projects")

    @property
    def user_publicshare_dir(self) -> str:
        return os.path.expanduser("~/Public")

    @property
    def user_templates_dir(self) -> str:
        return os.path.expanduser("~/Templates")

    @property
    def user_fonts_dir(self) -> str:
        return os.path.expanduser("~/Library/Fonts")

    @property
    def user_preference_dir(self) -> str:
        return self._with_app_parts(os.path.expanduser("~/Library/Preferences"))

    @property
    def user_bin_dir(self) -> str:
        return os.path.expanduser("~/.local/bin")

    @property
    def site_bin_dir(self) -> str:
        return "/usr/local/bin"

    @property
    def user_applications_dir(self) -> str:
        return os.path.expanduser("~/Applications")

    @property
    def _site_applications_dirs(self) -> list[str]:
        return ["/Applications"]

    @property
    def site_applications_dir(self) -> str:
        directories = self._site_applications_dirs
        if self.multipath:
            return os.pathsep.join(directories)
        return directories[0]

    @property
    def user_runtime_dir(self) -> str:
        return self._with_app_parts(os.path.expanduser("~/Library/Caches/TemporaryItems"))

    @property
    def site_runtime_dir(self) -> str:
        return self.user_runtime_dir

    def _iter_config_dirs(self) -> Iterator[str]:
        yield self.user_config_dir
        yield from self._site_config_dirs

    def _iter_data_dirs(self) -> Iterator[str]:
        yield self.user_data_dir
        yield from self._site_data_dirs

    def _iter_cache_dirs(self) -> Iterator[str]:
        yield self.user_cache_dir
        yield from self._site_cache_dirs


class MacOS(XDGMixin, _MacOSDefaults):
    pass