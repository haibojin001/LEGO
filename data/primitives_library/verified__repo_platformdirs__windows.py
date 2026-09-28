from __future__ import annotations

import os
import sys
import uuid
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .api import PlatformDirsABC

if TYPE_CHECKING:
    from collections.abc import Callable

_KF_FLAG_DONT_VERIFY: Final[int] = 0x00004000


class Windows(PlatformDirsABC):
    @property
    def user_data_dir(self) -> str:
        folder = "CSIDL_APPDATA" if self.roaming else "CSIDL_LOCAL_APPDATA"
        return self._append_parts(os.path.normpath(get_win_folder(folder)))

    def _append_parts(self, path: str, *, opinion_value: str | None = None) -> str:
        pieces: list[str] = []
        if self.appname:
            if self.appauthor is not False:
                pieces.append(self.appauthor or self.appname)
            pieces.append(self.appname)
            if opinion_value is not None and self.opinion:
                pieces.append(opinion_value)
            if self.version:
                pieces.append(self.version)
        result = os.path.join(path, *pieces)
        self._optionally_create_directory(result)
        return result

    @property
    def site_data_dir(self) -> str:
        return self._append_parts(os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA")))

    @property
    def user_config_dir(self) -> str:
        return self.user_data_dir

    @property
    def site_config_dir(self) -> str:
        return self.site_data_dir

    @property
    def user_cache_dir(self) -> str:
        root = os.path.normpath(get_win_folder("CSIDL_LOCAL_APPDATA"))
        return self._append_parts(root, opinion_value="Cache")

    @property
    def site_cache_dir(self) -> str:
        root = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
        return self._append_parts(root, opinion_value="Cache")

    @property
    def user_state_dir(self) -> str:
        return self.user_data_dir

    @property
    def site_state_dir(self) -> str:
        return self.site_data_dir

    @property
    def user_log_dir(self) -> str:
        result = self.user_data_dir
        if self.opinion:
            result = os.path.join(result, "Logs")
            self._optionally_create_directory(result)
        return result

    @property
    def site_log_dir(self) -> str:
        result = self.site_data_dir
        if self.opinion:
            result = os.path.join(result, "Logs")
            self._optionally_create_directory(result)
        return result

    @property
    def user_documents_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_PERSONAL"))

    @property
    def user_downloads_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_DOWNLOADS"))

    @property
    def user_pictures_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_MYPICTURES"))

    @property
    def user_videos_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_MYVIDEO"))

    @property
    def user_music_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_MYMUSIC"))

    @property
    def user_desktop_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_DESKTOPDIRECTORY"))

    @property
    def user_projects_dir(self) -> str:
        return os.path.normpath(os.path.expanduser("~/Projects"))

    @property
    def user_publicshare_dir(self) -> str:
        fallback = str(Path("~").expanduser().parent / "Public")
        return os.path.normpath(os.environ.get("PUBLIC", fallback))

    @property
    def user_templates_dir(self) -> str:
        path = Path(get_win_folder("CSIDL_APPDATA")) / "Microsoft" / "Windows" / "Templates"
        return os.path.normpath(str(path))

    @property
    def user_fonts_dir(self) -> str:
        path = Path(get_win_folder("CSIDL_LOCAL_APPDATA")) / "Microsoft" / "Windows" / "Fonts"
        return os.path.normpath(str(path))

    @property
    def user_preference_dir(self) -> str:
        return self.user_config_dir

    @property
    def user_bin_dir(self) -> str:
        return os.path.normpath(os.path.join(get_win_folder("CSIDL_LOCAL_APPDATA"), "Programs"))

    @property
    def site_bin_dir(self) -> str:
        return os.path.normpath(os.path.join(get_win_folder("CSIDL_COMMON_APPDATA"), "bin"))

    @property
    def user_applications_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_PROGRAMS"))

    @property
    def site_applications_dir(self) -> str:
        return os.path.normpath(get_win_folder("CSIDL_COMMON_PROGRAMS"))

    @property
    def user_runtime_dir(self) -> str:
        root = os.path.normpath(os.path.join(get_win_folder("CSIDL_LOCAL_APPDATA"), "Temp"))
        return self._append_parts(root)

    @property
    def site_runtime_dir(self) -> str:
        return self.user_runtime_dir


def get_win_folder_from_env_vars(csidl_name: str) -> str:
    value = get_win_folder_if_csidl_name_not_env_var(csidl_name)
    if value is not None:
        return value

    variable = {
        "CSIDL_APPDATA": "APPDATA",
        "CSIDL_COMMON_APPDATA": "ALLUSERSPROFILE",
        "CSIDL_LOCAL_APPDATA": "LOCALAPPDATA",
    }.get(csidl_name)
    if variable is None:
        raise ValueError(f"Unknown CSIDL name: {csidl_name}")

    value = os.environ.get(variable)
    if value is None:
        raise ValueError(f"Unset environment variable: {variable}")
    return value


def get_win_folder_if_csidl_name_not_env_var(csidl_name: str) -> str | None:
    home = os.path.expanduser("~")
    suffixes = {
        "CSIDL_PERSONAL": ("Documents",),
        "CSIDL_DOWNLOADS": ("Downloads",),
        "CSIDL_MYPICTURES": ("Pictures",),
        "CSIDL_MYVIDEO": ("Videos",),
        "CSIDL_MYMUSIC": ("Music",),
        "CSIDL_DESKTOPDIRECTORY": ("Desktop",),
    }
    suffix = suffixes.get(csidl_name)
    if suffix is not None:
        return os.path.join(home, *suffix)

    if csidl_name == "CSIDL_PROGRAMS":
        appdata = os.environ.get("APPDATA")
        if appdata is not None:
            return os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs")
        return None

    if csidl_name == "CSIDL_COMMON_PROGRAMS":
        common = os.environ.get("ALLUSERSPROFILE")
        if common is not None:
            return os.path.join(common, "Microsoft", "Windows", "Start Menu", "Programs")
        return None

    return None


def get_win_folder_from_registry(csidl_name: str) -> str:
    import winreg

    registry_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
    }.get(csidl_name)

    if registry_name is None:
        return get_win_folder_from_env_vars(csidl_name)

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            value, _ = winreg.QueryValueEx(key, registry_name)
    except OSError:
        return get_win_folder_from_env_vars(csidl_name)

    return str(value)


def get_win_folder_from_ctypes(csidl_name: str) -> str:
    import ctypes

    known_folders = {
        "CSIDL_APPDATA": "3EB685DB-65F9-4CF6-A03A-E3EF65729F3D",
        "CSIDL_COMMON_APPDATA": "62AB5D82-FDC1-4DC3-A9DD-070D1D495D97",
        "CSIDL_LOCAL_APPDATA": "F1B32785-6FBA-4FCF-9D55-7B8E7F157091",
        "CSIDL_PERSONAL": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
        "CSIDL_DOWNLOADS": "374DE290-123F-4565-9164-39C4925E467B",
        "CSIDL_MYPICTURES": "33E28130-4E1E-4676-835A-98395C3BC3BB",
        "CSIDL_MYVIDEO": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
        "CSIDL_MYMUSIC": "4BD8D571-6D19-48D3-BE97-422220080E43",
        "CSIDL_DESKTOPDIRECTORY": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
        "CSIDL_PROGRAMS": "A77F5D77-2E2B-44C3-A6A2-ABA601054A51",
        "CSIDL_COMMON_PROGRAMS": "0139D44E-6AFE-49F2-8690-3DAFCAE6FFB8",
    }

    try:
        identifier = known_folders[csidl_name]
    except KeyError as exc:
        raise ValueError(f"Unknown CSIDL name: {csidl_name}") from exc

    folder_id = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID(identifier).bytes_le)
    output = ctypes.c_wchar_p()
    result = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(folder_id),
        _KF_FLAG_DONT_VERIFY,
        None,
        ctypes.byref(output),
    )
    if result != 0:
        raise ctypes.WinError(result)

    try:
        return str(output.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(output)


@cache
def get_win_folder(csidl_name: str) -> str:
    if sys.platform == "win32":
        try:
            return get_win_folder_from_ctypes(csidl_name)
        except ImportError:
            return get_win_folder_from_registry(csidl_name)
    return get_win_folder_from_env_vars(csidl_name)