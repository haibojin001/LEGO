import logging
from datetime import datetime

try:
    import _winreg as winreg
except ImportError:
    import winreg

import zoneinfo

from tzlocal import utils
from tzlocal.windows_tz import win_tz


_cache_tz = None
_cache_tz_name = None

log = logging.getLogger("tzlocal")


def valuestodict(key):
    """Convert a registry key's values to a dictionary."""
    result = {}
    size = winreg.QueryInfoKey(key)[1]

    for index in range(size):
        data = winreg.EnumValue(key, index)
        result[data[0]] = data[1]

    return result


def _get_dst_info(tz):
    # Find the offset for when it doesn't have DST.
    dst_offset = std_offset = None
    has_dst = False
    year = datetime.now().year

    for dt in (datetime(year, 1, 1), datetime(year, 6, 1)):
        if tz.dst(dt).total_seconds() == 0.0:
            std_offset = tz.utcoffset(dt).total_seconds()
        else:
            has_dst = True

    return has_dst, std_offset, dst_offset


def _get_localzone_name():
    # Windows timezone names may be localized, so map the registry
    # timezone key back to an IANA zone name.
    tzenv = utils._tz_name_from_env()
    if tzenv:
        return tzenv

    log.debug("Looking up time zone info from registry")
    handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)

    timezone_key = winreg.OpenKey(
        handle,
        r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation",
    )
    keyvalues = valuestodict(timezone_key)
    timezone_key.Close()

    if "TimeZoneKeyName" not in keyvalues:
        raise LookupError("Can not find Windows timezone configuration")

    tzkeyname = keyvalues["TimeZoneKeyName"].split("\x00", 1)[0]

    timezone = win_tz.get(tzkeyname)
    if timezone is None:
        timezone = win_tz.get(tzkeyname + " Standard Time")

    if timezone is None:
        raise zoneinfo.ZoneInfoNotFoundError(tzkeyname)

    if keyvalues.get("DynamicDaylightTimeDisabled", 0) == 1:
        tz = zoneinfo.ZoneInfo(timezone)
        has_dst, std_offset, dst_offset = _get_dst_info(tz)

        if not has_dst:
            return timezone

        if std_offset is None:
            raise zoneinfo.ZoneInfoNotFoundError(
                f"{tzkeyname} claims to not have a non-DST time!?"
            )

        if std_offset % 3600:
            raise zoneinfo.ZoneInfoNotFoundError(
                f"tzlocal can't support disabling DST in the {timezone} zone."
            )

        return f"Etc/GMT{-std_offset // 3600:+.0f}"

    return timezone


def get_localzone_name() -> str:
    """Get the zoneinfo timezone name that matches the Windows-configured timezone."""
    global _cache_tz_name

    if _cache_tz_name is None:
        _cache_tz_name = _get_localzone_name()

    return _cache_tz_name


def get_localzone() -> zoneinfo.ZoneInfo:
    """Returns the zoneinfo-based tzinfo object that matches the Windows-configured timezone."""
    global _cache_tz

    if _cache_tz is None:
        _cache_tz = zoneinfo.ZoneInfo(get_localzone_name())

    if not utils._tz_name_from_env():
        utils.assert_tz_offset(_cache_tz, error=False)

    return _cache_tz


def reload_localzone() -> zoneinfo.ZoneInfo:
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz
    global _cache_tz_name

    _cache_tz_name = _get_localzone_name()
    _cache_tz = zoneinfo.ZoneInfo(_cache_tz_name)
    utils.assert_tz_offset(_cache_tz, error=False)

    return _cache_tz