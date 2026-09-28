from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from .api import PlatformDirsABC


class Android(PlatformDirsABC):
    """Platform directory provider for Android systems."""

    @property
    def user_data_dir(self) -> str:
        return self._append_app_name_and_version(cast("str", _android_folder()), "files")

    @property
    def site_data_dir(self) -> str:
        return self.user_data_dir

    @property
    def user_config_dir(self) -> str:
        return self._append_app_name_and_version(cast("str", _android_folder()), "shared_prefs")

    @property
    def site_config_dir(self) -> str:
        return self.user_config_dir

    @property
    def user_cache_dir(self) -> str:
        return self._append_app_name_and_version(cast("str", _android_folder()), "cache")

    @property
    def site_cache_dir(self) -> str:
        return self.user_cache_dir

    @property
    def user_state_dir(self) -> str:
        return self.user_data_dir

    @property
    def site_state_dir(self) -> str:
        return self.user_state_dir

    @property
    def user_log_dir(self) -> str:
        path = self.user_cache_dir
        if self.opinion:
            path = os.path.join(path, "log")
            self._optionally_create_directory(path)
        return path

    @property
    def site_log_dir(self) -> str:
        return self.user_log_dir

    @property
    def user_documents_dir(self) -> str:
        return _android_documents_folder()

    @property
    def user_downloads_dir(self) -> str:
        return _android_downloads_folder()

    @property
    def user_pictures_dir(self) -> str:
        return _android_pictures_folder()

    @property
    def user_videos_dir(self) -> str:
        return _android_videos_folder()

    @property
    def user_music_dir(self) -> str:
        return _android_music_folder()

    @property
    def user_desktop_dir(self) -> str:
        return "/storage/emulated/0/Desktop"

    @property
    def user_projects_dir(self) -> str:
        return "/storage/emulated/0/Projects"

    @property
    def user_publicshare_dir(self) -> str:
        return "/storage/emulated/0/Public"

    @property
    def user_templates_dir(self) -> str:
        return "/storage/emulated/0/Templates"

    @property
    def user_fonts_dir(self) -> str:
        return "/storage/emulated/0/fonts"

    @property
    def user_preference_dir(self) -> str:
        return self.user_config_dir

    @property
    def user_bin_dir(self) -> str:
        return os.path.join(cast("str", _android_folder()), "files", "bin")

    @property
    def site_bin_dir(self) -> str:
        return self.user_bin_dir

    @property
    def user_applications_dir(self) -> str:
        return self.user_data_dir

    @property
    def site_applications_dir(self) -> str:
        return self.user_applications_dir

    @property
    def user_runtime_dir(self) -> str:
        path = self.user_cache_dir
        if self.opinion:
            path = os.path.join(path, "tmp")
            self._optionally_create_directory(path)
        return path

    @property
    def site_runtime_dir(self) -> str:
        return self.user_runtime_dir


@lru_cache(maxsize=1)
def _android_folder() -> str | None:
    result: str | None = None

    if not TYPE_CHECKING:
        try:
            from android import mActivity

            context = cast("android.content.Context", mActivity.getApplicationContext())
            result = context.getFilesDir().getParentFile().getAbsolutePath()
        except Exception:
            result = None

    if result is None:
        try:
            from jnius import autoclass

            context = autoclass("android.content.Context")
            result = context.getFilesDir().getParentFile().getAbsolutePath()
        except Exception:
            result = None

    if result is None:
        pattern = re.compile(r"/data/(data|user/\d+)/(.+)/files")
        for path in sys.path:
            if pattern.match(path):
                result = path.split("/files")[0]
                break
        else:
            result = None

    if result is None:
        pattern = re.compile(r"/mnt/expand/[a-fA-F0-9-]{36}/(data|user/\d+)/(.+)/files")
        for path in sys.path:
            if pattern.match(path):
                result = path.split("/files")[0]
                break
        else:
            result = None

    return result


@lru_cache(maxsize=1)
def _android_documents_folder() -> str:
    try:
        from jnius import autoclass

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        result: str = context.getExternalFilesDir(environment.DIRECTORY_DOCUMENTS).getAbsolutePath()
    except Exception:
        result = "/storage/emulated/0/Documents"
    return result


@lru_cache(maxsize=1)
def _android_downloads_folder() -> str:
    try:
        from jnius import autoclass

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        result: str = context.getExternalFilesDir(environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
    except Exception:
        result = "/storage/emulated/0/Downloads"
    return result


@lru_cache(maxsize=1)
def _android_pictures_folder() -> str:
    try:
        from jnius import autoclass

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        result: str = context.getExternalFilesDir(environment.DIRECTORY_PICTURES).getAbsolutePath()
    except Exception:
        result = "/storage/emulated/0/Pictures"
    return result


@lru_cache(maxsize=1)
def _android_videos_folder() -> str:
    try:
        from jnius import autoclass

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        result: str = context.getExternalFilesDir(environment.DIRECTORY_MOVIES).getAbsolutePath()
    except Exception:
        result = "/storage/emulated/0/DCIM/Camera"
    return result


@lru_cache(maxsize=1)
def _android_music_folder() -> str:
    try:
        from jnius import autoclass

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        result: str = context.getExternalFilesDir(environment.DIRECTORY_MUSIC).getAbsolutePath()
    except Exception:
        result = "/storage/emulated/0/Music"
    return result