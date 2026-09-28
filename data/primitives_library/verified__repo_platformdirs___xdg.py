from __future__ import annotations

import os

from .api import PlatformDirsABC


def _xdg_dir_list(env_var: str) -> list[str]:
    value = os.environ.get(env_var, "")
    return [entry for item in value.split(os.pathsep) if (entry := item.strip())]


class XDGMixin(PlatformDirsABC):
    @property
    def user_data_dir(self) -> str:
        value = os.environ.get("XDG_DATA_HOME", "").strip()
        if value:
            return self._append_app_name_and_version(value)
        return super().user_data_dir

    @property
    def _site_data_dirs(self) -> list[str]:
        values = _xdg_dir_list("XDG_DATA_DIRS")
        if values:
            return [self._append_app_name_and_version(value) for value in values]
        return super()._site_data_dirs

    @property
    def site_data_dir(self) -> str:
        values = self._site_data_dirs
        return os.pathsep.join(values) if self.multipath else values[0]

    @property
    def user_config_dir(self) -> str:
        value = os.environ.get("XDG_CONFIG_HOME", "").strip()
        if value:
            return self._append_app_name_and_version(value)
        return super().user_config_dir

    @property
    def _site_config_dirs(self) -> list[str]:
        values = _xdg_dir_list("XDG_CONFIG_DIRS")
        if values:
            return [self._append_app_name_and_version(value) for value in values]
        return super()._site_config_dirs

    @property
    def site_config_dir(self) -> str:
        values = self._site_config_dirs
        return os.pathsep.join(values) if self.multipath else values[0]

    @property
    def user_cache_dir(self) -> str:
        value = os.environ.get("XDG_CACHE_HOME", "").strip()
        if value:
            return self._append_app_name_and_version(value)
        return super().user_cache_dir

    @property
    def user_state_dir(self) -> str:
        value = os.environ.get("XDG_STATE_HOME", "").strip()
        if value:
            return self._append_app_name_and_version(value)
        return super().user_state_dir

    @property
    def user_runtime_dir(self) -> str:
        value = os.environ.get("XDG_RUNTIME_DIR", "").strip()
        if value:
            return self._append_app_name_and_version(value)
        return super().user_runtime_dir

    @property
    def site_runtime_dir(self) -> str:
        value = os.environ.get("XDG_RUNTIME_DIR", "").strip()
        if value:
            return self._append_app_name_and_version(value)
        return super().site_runtime_dir

    @property
    def user_documents_dir(self) -> str:
        value = os.environ.get("XDG_DOCUMENTS_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_documents_dir

    @property
    def user_downloads_dir(self) -> str:
        value = os.environ.get("XDG_DOWNLOAD_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_downloads_dir

    @property
    def user_pictures_dir(self) -> str:
        value = os.environ.get("XDG_PICTURES_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_pictures_dir

    @property
    def user_videos_dir(self) -> str:
        value = os.environ.get("XDG_VIDEOS_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_videos_dir

    @property
    def user_music_dir(self) -> str:
        value = os.environ.get("XDG_MUSIC_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_music_dir

    @property
    def user_desktop_dir(self) -> str:
        value = os.environ.get("XDG_DESKTOP_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_desktop_dir

    @property
    def user_projects_dir(self) -> str:
        value = os.environ.get("XDG_PROJECTS_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_projects_dir

    @property
    def user_publicshare_dir(self) -> str:
        value = os.environ.get("XDG_PUBLICSHARE_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_publicshare_dir

    @property
    def user_templates_dir(self) -> str:
        value = os.environ.get("XDG_TEMPLATES_DIR", "").strip()
        if value:
            return os.path.expanduser(value)
        return super().user_templates_dir

    @property
    def user_fonts_dir(self) -> str:
        value = os.environ.get("XDG_DATA_HOME", "").strip()
        if value:
            return f"{os.path.expanduser(value)}/fonts"
        return super().user_fonts_dir

    @property
    def user_applications_dir(self) -> str:
        value = os.environ.get("XDG_DATA_HOME", "").strip()
        if value:
            return os.path.join(os.path.expanduser(value), "applications")
        return super().user_applications_dir

    @property
    def _site_applications_dirs(self) -> list[str]:
        values = _xdg_dir_list("XDG_DATA_DIRS")
        if values:
            return [os.path.join(value, "applications") for value in values]
        return super()._site_applications_dirs

    @property
    def site_applications_dir(self) -> str:
        values = self._site_applications_dirs
        return os.pathsep.join(values) if self.multipath else values[0]


__all__ = ["XDGMixin"]