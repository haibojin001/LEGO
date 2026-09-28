from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal


class PlatformDirsABC(ABC):
    def __init__(
        self,
        appname: str | None = None,
        appauthor: str | Literal[False] | None = None,
        version: str | None = None,
        roaming: bool = False,
        multipath: bool = False,
        opinion: bool = True,
        ensure_exists: bool = False,
        use_site_for_root: bool = False,
    ) -> None:
        self.appname = appname
        self.appauthor = appauthor
        self.version = version
        self.roaming = roaming
        self.multipath = multipath
        self.opinion = opinion
        self.ensure_exists = ensure_exists
        self.use_site_for_root = use_site_for_root

    def _append_app_name_and_version(self, *base: str) -> str:
        parts = list(base[1:])
        if self.appname:
            parts.append(self.appname)
            if self.version:
                parts.append(self.version)
        result = os.path.join(base[0], *parts)
        self._optionally_create_directory(result)
        return result

    def _optionally_create_directory(self, path: str) -> None:
        if self.ensure_exists:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _first_item_as_path_if_multipath(self, directory: str) -> Path:
        if self.multipath:
            directory = directory.partition(os.pathsep)[0]
        return Path(directory)

    @property
    @abstractmethod
    def user_data_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_data_dir(self) -> str:
        raise NotImplementedError

    @property
    def _site_data_dirs(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_config_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_config_dir(self) -> str:
        raise NotImplementedError

    @property
    def _site_config_dirs(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_cache_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_cache_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_state_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_state_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_log_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_log_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_documents_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_downloads_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_pictures_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_videos_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_music_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_desktop_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_projects_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_publicshare_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_templates_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_fonts_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_preference_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_bin_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_bin_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_applications_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_applications_dir(self) -> str:
        raise NotImplementedError

    @property
    def _site_applications_dirs(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def user_runtime_dir(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def site_runtime_dir(self) -> str:
        raise NotImplementedError

    @property
    def user_data_path(self) -> Path:
        return Path(self.user_data_dir)

    @property
    def site_data_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_data_dir)

    @property
    def site_data_paths(self) -> Iterator[Path]:
        return map(Path, self._site_data_dirs)

    @property
    def user_config_path(self) -> Path:
        return Path(self.user_config_dir)

    @property
    def site_config_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_config_dir)

    @property
    def site_config_paths(self) -> Iterator[Path]:
        return map(Path, self._site_config_dirs)

    @property
    def user_cache_path(self) -> Path:
        return Path(self.user_cache_dir)

    @property
    def site_cache_path(self) -> Path:
        return Path(self.site_cache_dir)

    @property
    def user_state_path(self) -> Path:
        return Path(self.user_state_dir)

    @property
    def site_state_path(self) -> Path:
        return Path(self.site_state_dir)

    @property
    def user_log_path(self) -> Path:
        return Path(self.user_log_dir)

    @property
    def site_log_path(self) -> Path:
        return Path(self.site_log_dir)

    @property
    def user_documents_path(self) -> Path:
        return Path(self.user_documents_dir)

    @property
    def user_downloads_path(self) -> Path:
        return Path(self.user_downloads_dir)

    @property
    def user_pictures_path(self) -> Path:
        return Path(self.user_pictures_dir)

    @property
    def user_videos_path(self) -> Path:
        return Path(self.user_videos_dir)

    @property
    def user_music_path(self) -> Path:
        return Path(self.user_music_dir)

    @property
    def user_desktop_path(self) -> Path:
        return Path(self.user_desktop_dir)

    @property
    def user_projects_path(self) -> Path:
        return Path(self.user_projects_dir)

    @property
    def user_publicshare_path(self) -> Path:
        return Path(self.user_publicshare_dir)

    @property
    def user_templates_path(self) -> Path:
        return Path(self.user_templates_dir)

    @property
    def user_fonts_path(self) -> Path:
        return Path(self.user_fonts_dir)

    @property
    def user_preference_path(self) -> Path:
        return Path(self.user_preference_dir)

    @property
    def user_bin_path(self) -> Path:
        return Path(self.user_bin_dir)

    @property
    def site_bin_path(self) -> Path:
        return Path(self.site_bin_dir)

    @property
    def user_applications_path(self) -> Path:
        return Path(self.user_applications_dir)

    @property
    def site_applications_path(self) -> Path:
        return self._first_item_as_path_if_multipath(self.site_applications_dir)

    @property
    def site_applications_paths(self) -> Iterator[Path]:
        return map(Path, self._site_applications_dirs)

    @property
    def user_runtime_path(self) -> Path:
        return Path(self.user_runtime_dir)

    @property
    def site_runtime_path(self) -> Path:
        return Path(self.site_runtime_dir)