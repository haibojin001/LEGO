from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from typing import IO, TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    import psutil


__version__ = "1.5.0"

_DEFAULT = object()
POSIX = os.name == "posix"
ENCODING = sys.getfilesystemencoding()

try:
    ENCODING_ERRS = sys.getfilesystemencodeerrors()
except AttributeError:
    ENCODING_ERRS = "surrogateescape" if POSIX else "replace"


def get_processor_name() -> str:
    """Returns the processor name in the system."""
    system = platform.system()

    if system == "Linux":
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if "model name" in line:
                    return re.sub(r".*model name.*:", "", line, count=1)

        model = cat("/proc/device-tree/model", fallback=b"", binary=True)
        name = model.split(b"\x00", 1)[0].decode(errors="replace").strip()
        if name:
            return name

    elif system == "FreeBSD":
        return subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True
        ).strip()

    elif system == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()

    return platform.processor()


def kill_child_processes(parent_proc: psutil.Process | None, timeout: int = 3) -> None:
    """Kills a process and its descendants, escalating when needed."""
    import psutil

    if parent_proc is None:
        logging.debug("No stress process to kill")
        return

    logging.debug("Killing stress process %s", parent_proc)

    try:
        pgid = os.getpgid(parent_proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        logging.debug("Sent SIGTERM to process group %s", pgid)
    except (OSError, ProcessLookupError, AttributeError):
        logging.debug("Process group kill failed, falling back to per-process")
        try:
            for process in parent_proc.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess, ProcessLookupError):
                    process.terminate()
            parent_proc.terminate()
        except (psutil.NoSuchProcess, AttributeError):
            logging.debug("Process already gone during terminate")
            return

    try:
        _, alive = psutil.wait_procs(
            [parent_proc, *parent_proc.children(recursive=True)],
            timeout=timeout,
        )
        for process in alive:
            logging.debug("Sending SIGKILL to straggler %s", process)
            with contextlib.suppress(psutil.NoSuchProcess, ProcessLookupError):
                process.kill()
    except (psutil.NoSuchProcess, AttributeError):
        logging.debug("Process already gone during wait")


def _get_throttle_label(sources: Any) -> str:
    """Find the first nonempty throttle reason reported by available sources."""
    for source in sources:
        if not source.get_is_available():
            continue
        for suffix in source.get_sensor_suffixes():
            if suffix:
                return suffix
    return ""


def output_to_csv(sources: dict, csv_writeable_file: str) -> None:
    """Append source statistics to a CSV file."""
    file_exists = os.path.isfile(csv_writeable_file)

    with open(csv_writeable_file, "a") as csvfile:
        row = OrderedDict()
        row["Time"] = time.strftime("%Y-%m-%d_%H:%M:%S")

        summaries = [value for key, value in sources.items()]
        for summary in summaries:
            source = summary.source
            prefix = source.get_source_name() + ":"
            for property_name, value in source.get_sensors_summary().items():
                row[prefix + property_name] = value

        row["Throttle"] = _get_throttle_label([item.source for item in summaries])

        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def output_to_terminal(sources: list) -> None:
    """Emit source statistics in the regular terminal representation."""
    results = OrderedDict()

    for source in sources:
        if source.get_is_available():
            source.update()
            results[source.get_source_name()] = source.get_sensors_summary()

    throttle = _get_throttle_label(sources)
    if throttle:
        results["Throttle"] = {"reason": throttle}

    for name, values in results.items():
        sys.stdout.write(str(name) + ": ")
        for key, value in values.items():
            sys.stdout.write(str(key) + ": " + str(value) + ", ")

    sys.stdout.write("\n")
    sys.exit()


def output_to_json(sources: list) -> None:
    """Emit source statistics as formatted JSON."""
    results = OrderedDict()

    for source in sources:
        if source.get_is_available():
            source.update()
            results[source.get_source_name()] = source.get_sensors_summary()

    results["Throttle"] = _get_throttle_label(sources)
    print(json.dumps(results, indent=4))
    sys.exit()


def _get_xdg_config_home() -> str:
    """Return XDG's configuration home, or the conventional fallback."""
    config_home = os.getenv("XDG_CONFIG_HOME")
    if config_home:
        return config_home
    return os.path.expanduser(os.path.join("~", ".config"))


def get_config_dir() -> str:
    """Return the user's base configuration directory."""
    return _get_xdg_config_home()


def get_user_config_dir() -> str:
    """Return the s-tui-specific configuration directory."""
    return os.path.join(_get_xdg_config_home(), "s-tui")


def get_user_config_file() -> str:
    """Return the s-tui user configuration file path."""
    return os.path.join(get_user_config_dir(), "s-tui.conf")


def user_config_dir_exists() -> bool:
    """Tell whether the s-tui configuration directory exists."""
    return os.path.isdir(get_user_config_dir())


def config_dir_exists() -> bool:
    """Tell whether the base configuration directory exists."""
    return os.path.isdir(get_config_dir())


def user_config_file_exists() -> bool:
    """Tell whether the s-tui configuration file exists."""
    return os.path.isfile(get_user_config_file())


def make_user_config_dir() -> str | None:
    """Create the needed s-tui configuration directories when absent."""
    config_dir = get_config_dir()
    config_path = get_user_config_dir()

    if not config_dir_exists():
        try:
            os.mkdir(config_dir)
        except OSError:
            return None

    if not user_config_dir_exists():
        try:
            os.mkdir(config_path)
            os.mkdir(os.path.join(config_path, "hooks.d"))
        except OSError:
            return None

    return config_path


def seconds_to_text(secs: float) -> str:
    """Format a duration in seconds as hours, minutes, and seconds."""
    hours = secs // 3600
    minutes = (secs - hours * 3600) // 60
    seconds = secs - hours * 3600 - minutes * 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def str_to_bool(string: str) -> bool:
    """Convert the supported boolean text values to booleans."""
    if string == "True":
        return True
    if string == "False":
        return False
    raise ValueError


def which(program: str) -> str | None:
    """Find the path of an executable."""
    def is_executable(path: str) -> bool:
        return os.path.isfile(path) and os.access(path, os.X_OK)

    path, filename = os.path.split(program)
    if path:
        if is_executable(program):
            return program
    else:
        for directory in os.environ["PATH"].split(os.pathsep):
            executable = os.path.join(directory, program)
            if is_executable(executable):
                return executable

    return None


def open_binary(fname: str, **kwargs: Any) -> IO[bytes]:
    """Open a file in binary read mode."""
    return open(fname, "rb", **kwargs)


def open_text(fname: str, **kwargs: Any) -> IO[str]:
    """Open a text file using the filesystem encoding defaults."""
    kwargs.setdefault("encoding", ENCODING)
    kwargs.setdefault("errors", ENCODING_ERRS)
    return open(fname, "rt", **kwargs)


@overload
def cat(
    fname: str,
    fallback: Any = _DEFAULT,
    binary: Literal[False] = False,
) -> str:
    ...


@overload
def cat(
    fname: str,
    fallback: Any = _DEFAULT,
    binary: Literal[True] = True,
) -> bytes:
    ...


def cat(
    fname: str,
    fallback: Any = _DEFAULT,
    binary: bool = False,
) -> str | bytes | Any:
    """Read a file, optionally returning a fallback when it cannot be read."""
    try:
        if binary:
            with open_binary(fname) as file:
                return file.read()
        with open_text(fname) as file:
            return file.read()
    except OSError:
        if fallback is _DEFAULT:
            raise
        return fallback