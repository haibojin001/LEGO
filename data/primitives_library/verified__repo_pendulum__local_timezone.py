from __future__ import annotations

import contextlib
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pendulum.tz.exceptions import InvalidTimezone
from pendulum.tz.timezone import UTC
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone

if TYPE_CHECKING:
    from collections.abc import Iterator

if sys.platform == "win32":
    import winreg

_mock_local_timezone = None
_local_timezone = None


def get_local_timezone() -> Timezone | FixedTimezone:
    global _local_timezone

    if _mock_local_timezone is not None:
        return _mock_local_timezone

    if _local_timezone is None:
        _local_timezone = _get_system_timezone()

    return _local_timezone


def set_local_timezone(mock: str | Timezone | None = None) -> None:
    global _mock_local_timezone

    _mock_local_timezone = mock


@contextmanager
def test_local_timezone(mock: Timezone) -> Iterator[None]:
    set_local_timezone(mock)
    yield
    set_local_timezone()


def _get_system_timezone() -> Timezone:
    if sys.platform == "win32":
        return _get_windows_timezone()

    if "darwin" in sys.platform:
        return _get_darwin_timezone()

    return _get_unix_timezone()


if sys.platform == "win32":

    def _get_windows_timezone() -> Timezone:
        from pendulum.tz.data.windows import windows_timezones

        registry = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        current_key = winreg.OpenKey(
            registry,
            r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation",
        )

        values = {}
        for index in range(winreg.QueryInfoKey(current_key)[1]):
            name, value, *_ = winreg.EnumValue(current_key, index)
            values[name] = value

        current_key.Close()

        if "TimeZoneKeyName" in values:
            key_name = values["TimeZoneKeyName"].split("\x00", 1)[0]
        else:
            standard_name = values["StandardName"]
            zones_key = winreg.OpenKey(
                registry,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones",
            )

            key_name = None
            for index in range(winreg.QueryInfoKey(zones_key)[0]):
                candidate = winreg.EnumKey(zones_key, index)
                candidate_key = winreg.OpenKey(zones_key, candidate)
                candidate_values = {}

                for value_index in range(winreg.QueryInfoKey(candidate_key)[1]):
                    name, value, *_ = winreg.EnumValue(candidate_key, value_index)
                    candidate_values[name] = value

                candidate_key.Close()

                with contextlib.suppress(KeyError):
                    if candidate_values["Std"] == standard_name:
                        key_name = candidate
                        break

            zones_key.Close()
            registry.Close()

        if key_name is None:
            raise LookupError("Can not find Windows timezone configuration")

        zone_name = windows_timezones.get(key_name)
        if zone_name is None:
            zone_name = windows_timezones.get(key_name + " Standard Time")

        if zone_name is None:
            raise LookupError("Unable to find timezone " + key_name)

        return Timezone(zone_name)

else:

    def _get_windows_timezone() -> Timezone:
        raise NotImplementedError


def _get_darwin_timezone() -> Timezone:
    destination = os.readlink("/etc/localtime")
    name = destination[destination.rfind("zoneinfo/") + 9 :]
    return Timezone(name)


def _get_unix_timezone(_root: str = "/") -> Timezone:
    environment_timezone = os.environ.get("TZ")
    if environment_timezone:
        with contextlib.suppress(ValueError):
            return _tz_from_env(environment_timezone)

    timezone_file = Path(_root) / "etc" / "timezone"
    if timezone_file.is_file():
        content = timezone_file.read_bytes()
        if not content.startswith(b"TZif2"):
            configured_name = content.strip().decode()
            configured_name, _, _ = configured_name.partition(" ")
            configured_name, _, _ = configured_name.partition("#")
            return Timezone(configured_name.replace(" ", "_"))

    assignment = re.compile(r'\s*(TIME)?ZONE\s*=\s*"([^"]+)?"')
    for relative_name in ("etc/sysconfig/clock", "etc/conf.d/clock"):
        clock_file = Path(_root) / relative_name
        if not clock_file.is_file():
            continue

        for line in clock_file.read_text().splitlines():
            match = assignment.match(line)
            if match is None:
                continue

            configured_name = match.group(2)
            remaining = list(
                reversed(configured_name.replace(" ", "_").split(os.path.sep))
            )
            name_parts: list[str] = []

            while remaining:
                name_parts.insert(0, remaining.pop(0))
                with contextlib.suppress(InvalidTimezone):
                    return Timezone(os.path.sep.join(name_parts))

    localtime = Path(_root) / "etc" / "localtime"
    if localtime.is_file() and localtime.is_symlink():
        remaining = [part.replace(" ", "_") for part in reversed(localtime.resolve().parts)]
        name_parts: list[str] = []

        while remaining:
            name_parts.insert(0, remaining.pop(0))
            with contextlib.suppress(InvalidTimezone):
                return Timezone(os.path.sep.join(name_parts))

    for relative_name in ("etc/localtime", "usr/local/etc/localtime"):
        localtime = Path(_root) / relative_name
        if localtime.is_file():
            with localtime.open("rb") as stream:
                return Timezone.from_file(stream)

    warnings.warn(
        "Unable not find any timezone configuration, defaulting to UTC.",
        stacklevel=1,
    )
    return UTC


def _tz_from_env(tzenv: str) -> Timezone:
    if tzenv[0] == ":":
        tzenv = tzenv[1:]

    if os.path.isfile(tzenv):
        with open(tzenv, "rb") as stream:
            return Timezone.from_file(stream)

    try:
        return Timezone(tzenv)
    except ValueError:
        raise