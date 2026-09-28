import logging
import os
import subprocess
import sys
from functools import lru_cache
import importlib.resources

from ._definitions import FNAME_PER_PLATFORM, get_platform


logger = logging.getLogger("imageio_ffmpeg")


def get_ffmpeg_exe():
    """Return the configured or discovered ffmpeg executable."""
    override = os.getenv("IMAGEIO_FFMPEG_EXE")
    if override:
        return override

    executable = _get_ffmpeg_exe()
    if executable:
        return executable

    raise RuntimeError(
        "No ffmpeg exe could be found. Install ffmpeg on your system, "
        "or set the IMAGEIO_FFMPEG_EXE environment variable."
    )


@lru_cache()
def _get_ffmpeg_exe():
    platform = get_platform()

    bundled = os.path.join(_get_bin_dir(), FNAME_PER_PLATFORM.get(platform, ""))
    if bundled and os.path.isfile(bundled) and _is_valid_exe(bundled):
        return bundled

    if platform.startswith("win"):
        conda_executable = os.path.join(
            sys.prefix, "Library", "bin", "ffmpeg.exe"
        )
    else:
        conda_executable = os.path.join(sys.prefix, "bin", "ffmpeg")

    if (
        conda_executable
        and os.path.isfile(conda_executable)
        and _is_valid_exe(conda_executable)
    ):
        return conda_executable

    if _is_valid_exe("ffmpeg"):
        return "ffmpeg"

    return None


def _get_bin_dir():
    if sys.version_info < (3, 9):
        resource_context = importlib.resources.path(
            "imageio_ffmpeg.binaries", "__init__.py"
        )
    else:
        resource = (
            importlib.resources.files("imageio_ffmpeg.binaries") / "__init__.py"
        )
        resource_context = importlib.resources.as_file(resource)

    with resource_context as resource_path:
        pass

    return str(resource_path.parent)


def _popen_kwargs(prevent_sigint=False):
    startupinfo = None
    preexec_fn = None
    creationflags = 0

    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    if prevent_sigint:
        if sys.platform.startswith("win"):
            creationflags = 0x00000200
        else:
            preexec_fn = os.setpgrp

    disabled_values = ("", "0", "false", "no")
    if (
        os.getenv("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "").lower()
        not in disabled_values
    ):
        preexec_fn = None

    return {
        "startupinfo": startupinfo,
        "creationflags": creationflags,
        "preexec_fn": preexec_fn,
    }


def _is_valid_exe(exe):
    try:
        with open(os.devnull, "w") as null_file:
            subprocess.check_call(
                [exe, "-version"],
                stdout=null_file,
                stderr=subprocess.STDOUT,
                **_popen_kwargs()
            )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    return True


def get_ffmpeg_version():
    """Return the version string reported by the selected ffmpeg executable."""
    executable = get_ffmpeg_exe()
    first_line = subprocess.check_output(
        [executable, "-version"], **_popen_kwargs()
    ).split(b"\n", 1)[0]
    text = first_line.decode(errors="ignore").strip()
    return text.split("version", 1)[-1].lstrip().split(" ", 1)[0].strip()