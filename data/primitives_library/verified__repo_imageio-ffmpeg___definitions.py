import platform
import sys

__version__ = "0.6.0"


def get_platform():
    return _get_os_string() + "-" + _get_arch()


def _get_os_string():
    current = sys.platform
    if current.startswith("win"):
        return "windows"
    if current.startswith("darwin"):
        return "macos"
    if current.startswith("linux"):
        return "linux"
    return current


def _get_arch():
    machine = platform.machine()
    is_64bit = sys.maxsize > 2**32

    if machine == "armv7l":
        return "armv7"
    if is_64bit and machine.startswith(("arm", "aarch64")):
        return "aarch64"
    if is_64bit:
        return "x86_64"
    return "i686"


FNAME_PER_PLATFORM = {
    "macos-aarch64": "ffmpeg-macos-aarch64-v7.1",
    "macos-x86_64": "ffmpeg-macos-x86_64-v7.1",
    "windows-x86_64": "ffmpeg-win-x86_64-v7.1.exe",
    "windows-i686": "ffmpeg-win32-v4.2.2.exe",
    "linux-aarch64": "ffmpeg-linux-aarch64-v7.0.2",
    "linux-x86_64": "ffmpeg-linux-x86_64-v7.0.2",
}

osxplats = "macosx_10_9_intel.macosx_10_9_x86_64"
osxarmplats = "macosx_11_0_arm64"

WHEEL_BUILDS = {
    "py3-none-manylinux2014_x86_64": "linux-x86_64",
    "py3-none-manylinux2014_aarch64": "linux-aarch64",
    "py3-none-" + osxplats: "macos-x86_64",
    "py3-none-" + osxarmplats: "macos-aarch64",
    "py3-none-win32": "windows-i686",
    "py3-none-win_amd64": "windows-x86_64",
}