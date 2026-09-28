from __future__ import annotations

import os
import sys
from configparser import ConfigParser
from functools import cached_property
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, NoReturn

from ._xdg import XDGMixin
from .api import PlatformDirsABC

if TYPE_CHECKING:
    from collections.abc import Iterator

if sys.platform == "win32":

    def getuid() -> NoReturn:
        raise RuntimeError("should only be used on Unix")

else:
    from os import getuid


class _UnixDefaults(PlatformDirsABC):
    """Unix directory defaults without XDG environment overrides."""

    @cached_property
    def _use_site(self) -> bool:
        return self.use_site_for_root and getuid() == 0

    @property
    def user_data_dir(self) -> str:
        return self._append_app_name_and_version(os.path.expanduser("~/.local/share"))

    @property
    def _site_data_dirs(self) -> list[str]:
        return [
            self._append_app_name_and_version("/usr/local/share"),
            self._append_app_name_and_version("/usr/share"),
        ]

    @property
    def site_data_dir(self) -> str:
        directories = self._site_data_dirs
        return os.pathsep.join(directories) if self.multipath else directories[0]

    @property
    def user_config_dir(self) -> str:
        return self._append_app_name_and_version(os.path.expanduser("~/.config"))

    @property
    def _site_config_dirs(self) -> list[str]:
        return [self._append_app_name_and_version("/etc/xdg")]

    @property
    def site_config_dir(self) -> str:
        directories = self._site_config_dirs
        return os.pathsep.join(directories) if self.multipath else directories[0]

    @property
    def user_cache_dir(self) -> str:
        return self._append_app_name_and_version(os.path.expanduser("~/.cache"))

    @property
    def site_cache_dir(self) -> str:
        return self._append_app_name_and_version("/var/cache")

    @property
    def user_state_dir(self) -> str:
        return self._append_app_name_and_version(os.path.expanduser("~/.local/state"))

    @property
    def site_state_dir(self) -> str:
        return self._append_app_name_and_version("/var/lib")

    @property
    def user_log_dir(self) -> str:
        directory = self.user_state_dir
        if self.opinion:
            directory = os.path.join(directory, "log")
            self._optionally_create_directory(directory)
        return directory

    @property
    def site_log_dir(self) -> str:
        return self._append_app_name_and_version("/var/log")

    @property
    def user_documents_dir(self) -> str:
        return _get_user_media_dir("XDG_DOCUMENTS_DIR", "~/Documents")

    @property
    def user_downloads_dir(self) -> str:
        return _get_user_media_dir("XDG_DOWNLOAD_DIR", "~/Downloads")

    @property
    def user_pictures_dir(self) -> str:
        return _get_user_media_dir("XDG_PICTURES_DIR", "~/Pictures")

    @property
    def user_videos_dir(self) -> str:
        return _get_user_media_dir("XDG_VIDEOS_DIR", "~/Videos")

    @property
    def user_music_dir(self) -> str:
        return _get_user_media_dir("XDG_MUSIC_DIR", "~/Music")

    @property
    def user_desktop_dir(self) -> str:
        return _get_user_media_dir("XDG_DESKTOP_DIR", "~/Desktop")

    @property
    def user_projects_dir(self) -> str:
        return _get_user_media_dir("XDG_PROJECTS_DIR", "~/Projects")

    @property
    def user_publicshare_dir(self) -> str:
        return _get_user_media_dir("XDG_PUBLICSHARE_DIR", "~/Public")

    @property
    def user_templates_dir(self) -> str:
        return _get_user_media_dir("XDG_TEMPLATES_DIR", "~/Templates")

    @property
    def user_fonts_dir(self) -> str:
        return f"{os.path.expanduser('~/.local/share')}/fonts"

    @property
    def user_preference_dir(self) -> str:
        return self.user_config_dir

    @property
    def user_bin_dir(self) -> str:
        return os.path.expanduser("~/.local/bin")

    @property
    def site_bin_dir(self) -> str:
        return "/usr/local/bin"

    @property
    def user_applications_dir(self) -> str:
        return os.path.join(os.path.expanduser("~/.local/share"), "applications")

    @property
    def _site_applications_dirs(self) -> list[str]:
        return [
            os.path.join("/usr/local/share", "applications"),
            os.path.join("/usr/share", "applications"),
        ]

    @property
    def site_applications_dir(self) -> str:
        directories = self._site_applications_dirs
        return os.pathsep.join(directories) if self.multipath else directories[0]

    @property
    def user_runtime_dir(self) -> str:
        if sys.platform.startswith("openbsd"):
            directory = f"/tmp/run/user/{getuid()}"
        elif sys.platform.startswith(("freebsd", "netbsd")):
            directory = f"/var/run/user/{getuid()}"
        else:
            directory = f"/run/user/{getuid()}"

        if not os.access(directory, os.W_OK):
            directory = f"{gettempdir()}/runtime-{getuid()}"

        return self._append_app_name_and_version(directory)

    @property
    def site_runtime_dir(self) -> str:
        if sys.platform.startswith(("freebsd", "openbsd", "netbsd")):
            directory = "/var/run"
        else:
            directory = "/run"
        return self._append_app_name_and_version(directory)

    @property
    def site_data_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_data_dir)

    @property
    def site_config_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_config_dir)

    @property
    def site_cache_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_cache_dir)

    @property
    def site_state_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_state_dir)

    @property
    def site_log_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_log_dir)

    @property
    def site_runtime_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_runtime_dir)

    @property
    def site_applications_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_applications_dir)


class Unix(XDGMixin, _UnixDefaults):
    """Unix platform directories."""


def _get_user_media_dir(env_var: str, fallback: str) -> str:
    directory = next(
        (value for key, value in _get_user_dirs_values() if key == env_var),
        None,
    )
    return directory if directory is not None else os.path.expanduser(fallback)


def _get_user_dirs_values() -> Iterator[tuple[str, str]]:
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    user_dirs_file = Path(config_home) / "user-dirs.dirs"

    if not user_dirs_file.exists():
        return

    parser = ConfigParser(interpolation=None)
    parser.optionxform = str

    with user_dirs_file.open() as stream:
        parser.read_string("[user-dirs]\n" + stream.read())

    for key, value in parser["user-dirs"].items():
        yield key, os.path.expanduser(os.path.expandvars(value.strip('"')))