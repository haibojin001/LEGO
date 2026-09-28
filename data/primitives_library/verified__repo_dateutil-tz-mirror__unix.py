import logging
import os
import re
import warnings
import zoneinfo
from datetime import timezone

from tzlocal import utils

_cache_tz = None
_cache_tz_name = None

log = logging.getLogger("tzlocal")


def _get_localzone_name(_root="/"):
    tzenv = utils._tz_name_from_env()
    if tzenv:
        return tzenv

    getprop = os.path.join(_root, "system/bin/getprop")
    if os.path.exists(getprop):
        log.debug("This looks like Termux")
        import subprocess

        try:
            return subprocess.check_output(
                ["getprop", "persist.sys.timezone"]
            ).strip().decode()
        except (OSError, subprocess.CalledProcessError):
            log.debug("It's not termux?")

    configurations = {}

    for relative_path in ("etc/timezone", "var/db/zoneinfo"):
        path = os.path.join(_root, relative_path)
        try:
            with open(path, encoding="ascii") as handle:
                contents = handle.read()
        except (OSError, UnicodeDecodeError):
            continue

        log.debug(f"{path} found, contents:\n {contents}")
        contents = contents.strip("/ \t\r\n")
        if not contents:
            continue

        for item in contents.splitlines():
            if " " in item:
                item = item.split(" ", 1)[0]
            if "#" in item:
                item = item.split("#", 1)[0]
            if item:
                configurations[path] = item.replace(" ", "_")

    zone_pattern = re.compile(r'\s*ZONE\s*=\s*"')
    timezone_pattern = re.compile(r'\s*TIMEZONE\s*=\s*"')
    quote_pattern = re.compile('"')

    for relative_path in ("etc/sysconfig/clock", "etc/conf.d/clock"):
        path = os.path.join(_root, relative_path)
        try:
            with open(path, "rt") as handle:
                lines = handle.readlines()
        except (OSError, UnicodeDecodeError):
            continue

        log.debug(f"{path} found, contents:\n {lines}")

        for line in lines:
            match = zone_pattern.match(line)
            if match is None:
                match = timezone_pattern.match(line)

            if match is None:
                continue

            remainder = line[match.end() :]
            closing_quote = quote_pattern.search(remainder)
            if closing_quote is None:
                warnings.warn(f"Syntax error in {path}. Ignoring line: {line}")
                continue

            configurations[path] = remainder[: closing_quote.start()].replace(" ", "_")

    localtime_path = os.path.join(_root, "etc/localtime")
    if os.path.exists(localtime_path) and os.path.islink(localtime_path):
        log.debug(f"{localtime_path} found")
        target = os.path.realpath(localtime_path)

        offset = target.find("/") + 1
        while offset:
            candidate = target[offset:]
            try:
                zoneinfo.ZoneInfo(candidate)
            except zoneinfo.ZoneInfoNotFoundError:
                offset = candidate.find("/") + 1
                target = candidate
                continue

            configurations[f"{localtime_path} is a symlink to"] = candidate.replace(" ", "_")
            break

    if not configurations:
        return None

    log.debug(f"{len(configurations)} found:\n {configurations}")

    if len(configurations) > 1:
        unique_tzs = _get_unique_tzs(configurations, _root)

        if len(unique_tzs) != 1 and "etc/timezone" in str(configurations.keys()):
            log.warning(
                "/etc/timezone is deprecated in some distros, and no longer reliable. "
                "tzlocal is ignoring it, and you can likely delete it."
            )
            configurations = {
                path: name
                for path, name in configurations.items()
                if "etc/timezone" not in path
            }
            unique_tzs = _get_unique_tzs(configurations, _root)

        if len(unique_tzs) != 1:
            message = "Multiple conflicting time zone configurations found:\n"
            for path, name in configurations.items():
                message += f"{path}: {name}\n"
            message += (
                "Fix the configuration, or set the time zone in a TZ environment variable.\n"
            )
            raise zoneinfo.ZoneInfoNotFoundError(message)

    return list(configurations.values())[0]


def _get_unique_tzs(found_configs, _root):
    results = set()
    zoneinfo_root = os.path.join(_root, "usr", "share", "zoneinfo")
    prefix_parts = len(zoneinfo_root.split(os.path.sep))

    for name in found_configs.values():
        path = os.path.realpath(os.path.join(zoneinfo_root, *name.split("/")))
        results.add("/".join(path.split(os.path.sep)[prefix_parts:]))

    return results


def _get_localzone(_root="/"):
    environment_timezone = utils._tz_from_env()
    if environment_timezone:
        return environment_timezone

    name = _get_localzone_name(_root)

    if name is None:
        log.debug("No explicit setting existed. Use localtime")
        for relative_path in ("etc/localtime", "usr/local/etc/localtime"):
            path = os.path.join(_root, relative_path)
            if not os.path.exists(path):
                continue

            with open(path, "rb") as handle:
                tz = zoneinfo.ZoneInfo.from_file(handle, key="local")
            break
        else:
            warnings.warn(
                "Can not find any timezone configuration, defaulting to UTC."
            )
            utc_names = [
                item for item in zoneinfo.available_timezones() if "UTC" in item
            ]
            if utc_names:
                tz = zoneinfo.ZoneInfo(utc_names[0])
            else:
                tz = timezone.utc
    else:
        tz = zoneinfo.ZoneInfo(name)

    if _root == "/":
        utils.assert_tz_offset(tz, error=False)

    return tz


def get_localzone_name() -> str:
    global _cache_tz_name

    if _cache_tz_name is None:
        _cache_tz_name = _get_localzone_name()

    return _cache_tz_name


def get_localzone() -> zoneinfo.ZoneInfo:
    global _cache_tz

    if _cache_tz is None:
        _cache_tz = _get_localzone()

    return _cache_tz


def reload_localzone() -> zoneinfo.ZoneInfo:
    global _cache_tz
    global _cache_tz_name

    _cache_tz_name = _get_localzone_name()
    _cache_tz = _get_localzone()

    return _cache_tz