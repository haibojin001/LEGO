"""
babel.dates
~~~~~~~~~~~

Locale dependent formatting and parsing of dates and times.

The default locale for the functions in this module is determined by the
following environment variables, in that order:

 * ``LC_TIME``,
 * ``LC_ALL``, and
 * ``LANG``

:copyright: (c) 2013-2026 by the Babel Team.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, SupportsInt

try:
    import pytz
except ModuleNotFoundError:
    pytz = None
    import zoneinfo

import datetime
from collections.abc import Iterable

from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    _Instant: TypeAlias = datetime.date | datetime.time | float | None
    _PredefinedTimeFormat: TypeAlias = Literal['full', 'long', 'medium', 'short']
    _Context: TypeAlias = Literal['format', 'stand-alone']
    _DtOrTzinfo: TypeAlias = datetime.datetime | datetime.tzinfo | str | int | datetime.time | None  # fmt: skip

# "If a given short metazone form is known NOT to be understood in a given
#  locale and the parent locale has this value such that it would normally
#  be inherited, the inheritance of this value can be explicitly disabled by
#  use of the 'no inheritance marker' as the value, which is 3 simultaneous [sic]
#  empty set characters ( U+2205 )."
#  - https://www.unicode.org/reports/tr35/tr35-dates.html#Metazone_Names

NO_INHERITANCE_MARKER = '\u2205\u2205\u2205'

UTC = datetime.timezone.utc
LOCALTZ = localtime.LOCALTZ

LC_TIME = default_locale('LC_TIME')


def _localize(tz: datetime.tzinfo, dt: datetime.datetime) -> datetime.datetime:
    # Support localizing with both pytz and zoneinfo tzinfos
    # nothing to do
    if dt.tzinfo is tz:
        return dt

    if hasattr(tz, 'localize'):  # pytz
        return tz.localize(dt)

    if dt.tzinfo is None:
        # convert naive to localized
        return dt.replace(tzinfo=tz)

    # convert timezones
    return dt.astimezone(tz)


def _get_dt_and_tzinfo(
    dt_or_tzinfo: _DtOrTzinfo,
) -> tuple[datetime.datetime | None, datetime.tzinfo]:
    """
    Parse a `dt_or_tzinfo` value into a datetime and a tzinfo.

    See the docs for this function's callers for semantics.

    :rtype: tuple[datetime, tzinfo]
    """
    if dt_or_tzinfo is None:
        dt = datetime.datetime.now()
        tzinfo = LOCALTZ
    elif isinstance(dt_or_tzinfo, str):
        dt = None
        tzinfo = get_timezone(dt_or_tzinfo)
    elif isinstance(dt_or_tzinfo, int):
        dt = None
        tzinfo = UTC
    elif isinstance(dt_or_tzinfo, (datetime.datetime, datetime.time)):
        dt = _get_datetime(dt_or_tzinfo)
        tzinfo = dt.tzinfo if dt.tzinfo is not None else UTC
    else:
        dt = None
        tzinfo = dt_or_tzinfo
    return dt, tzinfo


def _get_tz_name(dt_or_tzinfo: _DtOrTzinfo) -> str:
    """
    Get the timezone name out of a time, datetime, or tzinfo object.

    :rtype: str
    """
    dt, tzinfo = _get_dt_and_tzinfo(dt_or_tzinfo)
    if hasattr(tzinfo, 'zone'):  # pytz object
        return tzinfo.zone
    elif hasattr(tzinfo, 'key') and tzinfo.key is not None:  # ZoneInfo object
        return tzinfo.key
    else:
        return tzinfo.tzname(dt or datetime.datetime.now(UTC))


def _get_datetime(instant: _Instant) -> datetime.datetime:
    """
    Get a datetime out of an "instant" (date, time, datetime, number).

    .. warning:: The return values of this function may depend on the system clock.

    If the instant is None, the current moment is used.
    If the instant is a time, it's augmented with today's date.

    Dates are converted to naive datetimes with midnight as the time component.

    >>> from datetime import date, datetime
    >>> _get_datetime(date(2015, 1, 1))
    datetime.datetime(2015, 1, 1, 0, 0)

    UNIX timestamps are converted to datetimes.

    >>> _get_datetime(1400000000)
    datetime.datetime(2014, 5, 13, 16, 53, 20)

    Other values are passed through as-is.

    >>> x = datetime(2015, 1, 1)
    >>> _get_datetime(x) is x
    True

    :param instant: date, time, datetime, integer, float or None
    :type instant: date|time|datetime|int|float|None
    :return: a datetime
    :rtype: datetime
    """
    if instant is None:
        return datetime.datetime.now(UTC).replace(tzinfo=None)
    elif isinstance(instant, (int, float)):
        return datetime.datetime.fromtimestamp(instant, UTC).replace(tzinfo=None)
    elif isinstance(instant, datetime.time):
        return datetime.datetime.combine(datetime.date.today(), instant)
    elif isinstance(instant, datetime.date) and not isinstance(instant, datetime.datetime):  # fmt: skip
        return datetime.datetime.combine(instant, datetime.time())
    # TODO (3.x): Add an assertion/type check for this fallthrough branch:
    return instant


def _ensure_datetime_tzinfo(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """
    Ensure the datetime passed has an attached tzinfo.

    If the datetime is tz-naive to begin with, UTC is attached.

    If a tzinfo is passed in, the datetime is normalized to that timezone.

    >>> from datetime import datetime
    >>> _get_tz_name(_ensure_datetime_tzinfo(datetime(2015, 1, 1)))
    'UTC'

    >>> tz = get_timezone("Europe/Stockholm")
    >>> _ensure_datetime_tzinfo(datetime(2015, 1, 1, 13, 15, tzinfo=UTC), tzinfo=tz).hour
    14

    :param datetime: Datetime to augment.
    :param tzinfo: optional tzinfo
    :return: datetime with tzinfo
    :rtype: datetime
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if tzinfo is not None:
        dt = dt.astimezone(get_timezone(tzinfo))
        if hasattr(tzinfo, 'normalize'):  # pytz
            dt = tzinfo.normalize(dt)
    return dt


def _get_time(
    time: datetime.time | datetime.datetime | None,
    tzinfo: datetime.tzinfo | None = None,
) -> datetime.time:
    """
    Get a timezoned time from a given instant.

    .. warning:: The return values of this function may depend on the system clock.

    :param time: time, datetime or None
    :rtype: time
    """
    if time is None:
        time = datetime.datetime.now(UTC)
    elif isinstance(time, (int, float)):
        time = datetime.datetime.fromtimestamp(time, UTC)

    if time.tzinfo is None:
        time = time.replace(tzinfo=UTC)

    if isinstance(time, datetime.datetime):
        if tzinfo is not None:
            time = time.astimezone(tzinfo)
            if hasattr(tzinfo, 'normalize'):  # pytz
                time = tzinfo.normalize(time)
        time = time.timetz()
    elif tzinfo is not None:
        time = time.replace(tzinfo=tzinfo)
    return time


def get_timezone(zone: str | datetime.tzinfo | None = None) -> datetime.tzinfo:
    """Looks up a timezone by name and returns it.  The timezone object
    returned comes from ``pytz`` or ``zoneinfo``, whichever is available.
    It corresponds to the `tzinfo` interface and can be used with all of
    the functions of Babel that operate with dates.

    If a timezone is not known a :exc:`LookupError` is raised.  If `zone`
    is ``None`` a local zone object is returned.

    :param zone: the name of the timezone to look up.  If a timezone object
                 itself is passed in, it's returned unchanged.
    """
    if zone is None:
        return LOCALTZ
    if not isinstance(zone, str):
        return zone

    if pytz:
        try:
            return pytz.timezone(zone)
        except pytz.UnknownTimeZoneError as e:
            exc = e
    else:
        assert zoneinfo
        try:
            return zoneinfo.ZoneInfo(zone)
        except zoneinfo.ZoneInfoNotFoundError as e:
            exc = e

    raise LookupError(f"Unknown timezone {zone}") from exc


def get_period_names(
    width: Literal['abbreviated', 'narrow', 'wide'] = 'wide',
    context: _Context = 'stand-alone',
    locale: Locale | str | None = None,
) -> LocaleDataDict:
    """Return the names for day periods (AM/PM) used by the locale.

    >>> get_period_names(locale='en_US')['am']
    'AM'

    :param width: the width to use, one of "abbreviated", "narrow", or "wide"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).day_periods[context][width]


def get_day_names(
    width: Literal['abbreviated', 'narrow', 'short', 'wide'] = 'wide',
    context: _Context = 'format',
    locale: Locale | str | None = None,
) -> LocaleDataDict:
    """Return the day names used by the locale for the specified format.

    >>> get_day_names('wide', locale='en_US')[1]
    'Tuesday'
    >>> get_day_names('short', locale='en_US')[1]
    'Tu'
    >>> get_day_names('abbreviated', locale='es')[1]
    'mar'
    >>> get_day_names('narrow', context='stand-alone', locale='de_DE')[1]
    'D'

    :param width: the width to use, one of "wide", "abbreviated", "short" or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).days[context][width]


def get_month_names(
    width: Literal['abbreviated', 'narrow', 'wide'] = 'wide',
    context: _Context = 'format',
    locale: Locale | str | None = None,
) -> LocaleDataDict:
    """Return the month names used by the locale for the specified format.

    >>> get_month_names('wide', locale='en_US')[1]
    'January'
    >>> get_month_names('abbreviated', locale='es')[1]
    'ene'
    >>> get_month_names('narrow', context='stand-alone', locale='de_DE')[1]
    'J'

    :param width: the width to use, one of "wide", "abbreviated", or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).months[context][width]


def get_quarter_names(
    width: Literal['abbreviated', 'narrow', 'wide'] = 'wide',
    context: _Context = 'format',
    locale: Locale | str | None = None,
) -> LocaleDataDict:
    """Return the quarter names used by the locale for the specified format.

    >>> get_quarter_names('wide', locale='en_US')[1]
    '1st quarter'
    >>> get_quarter_names('abbreviated', locale='de_DE')[1]
    'Q1'
    >>> get_quarter_names('narrow', locale='de_DE')[1]
    '1'

    :param width: the width to use, one of "wide", "abbreviated", or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).quarters[context][width]


def get_era_names(
    width: Literal['abbreviated', 'narrow', 'wide'] = 'wide',
    locale: Locale | str | None = None,
) -> LocaleDataDict:
    """Return the era names used by the locale for the specified format.

    >>> get_era_names('wide', locale='en_US')[1]
    'Anno Domini'
    >>> get_era_names('abbreviated', locale='de_DE')[1]
    'n. Chr.'

    :param width: the width to use, either "wide", "abbreviated", or "narrow"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).eras[width]


def get_date_format(
    format: _PredefinedTimeFormat = 'medium',
    locale: Locale | str | None = None,
) -> DateTimePattern:
    """Return the date formatting patterns used by the locale for the specified
    format.

    >>> get_date_format(locale='en_US')
    <DateTimePattern 'MMM d, y'>
    >>> get_date_format('full', locale='de_DE')
    <DateTimePattern 'EEEE, d. MMMM y'>

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).date_formats[format]


def get_datetime_format(
    format: _PredefinedTimeFormat = 'medium',
    locale: Locale | str | None = None,
) -> DateTimePattern:
    """Return the datetime formatting patterns used by the locale for the
    specified format.

    >>> get_datetime_format(locale='en_US')
    '{1}, {0}'

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    patterns = Locale.parse(locale or LC_TIME).datetime_formats
    if format not in patterns:
        format = None
    return patterns[format]


def get_time_format(
    format: _PredefinedTimeFormat = 'medium',
    locale: Locale | str | None = None,
) -> DateTimePattern:
    """Return the time formatting patterns used by the locale for the specified
    format.

    >>> get_time_format(locale='en_US')
    <DateTimePattern 'h:mm:ss\\u202fa'>
    >>> get_time_format('full', locale='de_DE')
    <DateTimePattern 'HH:mm:ss zzzz'>

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    """
    return Locale.parse(locale or LC_TIME).time_formats[format]


def get_timezone_gmt(
    datetime: _Instant = None,
    width: Literal['long', 'short', 'iso8601', 'iso8601_short'] = 'long',
    locale: Locale | str | None = None,
    return_z: bool = False,
) -> str:
    """Return the timezone associated with the given `datetime` object formatted
    as string indicating the offset from GMT.

    >>> from datetime import datetime
    >>> dt = datetime(2007, 4, 1, 15, 30)
    >>> get_timezone_gmt(dt, locale='en')
    'GMT+00:00'
    >>> get_timezone_gmt(dt, locale='en', return_z=True)
    'Z'
    >>> get_timezone_gmt(dt, locale='en', width='iso8601_short')
    '+00'
    >>> tz = get_timezone('America/Los_Angeles')
    >>> dt = _localize(tz, datetime(2007, 4, 1, 15, 30))
    >>> get_timezone_gmt(dt, locale='en')
    'GMT-07:00'
    >>> get_timezone_gmt(dt, 'short', locale='en')
    '-0700'
    >>> get_timezone_gmt(dt, locale='en', width='iso8601_short')
    '-07'

    The long format depends on the locale, for example in France the acronym
    UTC string is used instead of GMT:

    >>> get_timezone_gmt(dt, 'long', locale='fr_FR')
    'UTC-07:00'

    .. versionadded:: 0.9

    :param datetime: the ``datetime`` object; if `None`, the current date and
                     time in UTC is used
    :param width: either "long" or "short" or "iso8601" or "iso8601_short"
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    :param return_z: True or False; Function returns indicator "Z"
                     when local time offset is 0
    """
    datetime = _ensure_datetime_tzinfo(_get_datetime(datetime))
    locale = Locale.parse(locale or LC_TIME)

    offset = datetime.tzinfo.utcoffset(datetime)
    seconds = offset.days * 24 * 60 * 60 + offset.seconds
    hours, seconds = divmod(seconds, 3600)
    if return_z and hours == 0 and seconds == 0:
        return 'Z'
    elif seconds == 0 and width == 'iso8601_short':
        return '%+03d' % hours
    elif width == 'short' or width == 'iso8601_short':
        pattern = '%+03d%02d'
    elif width == 'iso8601':
        pattern = '%+03d:%02d'
    else:
        pattern = locale.zone_formats['gmt'] % '%+03d:%02d'
    return pattern % (hours, seconds // 60)


def get_timezone_location(
    dt_or_tzinfo: _DtOrTzinfo = None,
    locale: Locale | str | None = None,
    return_city: bool = False,
) -> str:
    """Return a representation of the given timezone using "location format".

    The result depends on both the local display name of the country and the
    city associated with the time zone:

    >>> tz = get_timezone('America/St_Johns')
    >>> print(get_timezone_location(tz, locale='de_DE'))
    Kanada (St. John’s) (Ortszeit)
    >>> print(get_timezone_location(tz, locale='en'))
    Canada (St. John’s) Time
    >>> print(get_timezone_location(tz, locale='en', return_city=True))
    St. John’s
    >>> tz = get_timezone('America/Mexico_City')
    >>> get_timezone_location(tz, locale='de_DE')
    'Mexiko (Mexiko-Stadt) (Ortszeit)'

    If the timezone is associated with a country that uses only a single
    timezone, just the localized country name is returned:

    >>> tz = get_timezone('Europe/Berlin')
    >>> get_timezone_name(tz, locale='de_DE')
    'Mitteleuropäische Zeit'

    .. versionadded:: 0.9

    :param dt_or_tzinfo: the ``datetime`` or ``tzinfo`` object that determines
                         the timezone; if `None`, the current date and time in
                         UTC is assumed
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    :param return_city: True or False, if True then return exemplar city (location)
                        for the time zone
    :return: the localized timezone name using location format

    """
    locale = Locale.parse(locale or LC_TIME)

    zone = _get_tz_name(dt_or_tzinfo)

    # Get the canonical time-zone code
    zone = get_global('zone_aliases').get(zone, zone)

    info = locale.time_zones.get(zone, {})

    # Otherwise, if there is only one timezone for the country, return the
    # localized country name
    region_format = locale.zone_formats['region']
    territory = get_global('zone_territories').get(zone)
    if territory not in locale.territories:
        territory = 'ZZ'  # invalid/unknown
    territory_name = locale.territories[territory]
    if (
        not return_city
        and territory
        and len(get_global('territory_zones').get(territory, [])) == 1
    ):
        return region_format % territory_name

    # Otherwise, include the city in the output
    fallback_format = locale.zone_formats['fallback']
    if 'city' in info:
        city_name = info['city']
    else:
        metazone = get_global('meta_zones').get(zone)
        metazone_info = locale.meta_zones.get(metazone, {})
        if 'city' in metazone_info:
            city_name = metazone_info['city']
        elif '/' in zone:
            city_name = zone.split('/', 1)[1].replace('_', ' ')
        else:
            city_name = zone.replace('_', ' ')

    if return_city:
        return city_name
    return region_format % (
        fallback_format
        % {
            '0': city_name,
            '1': territory_name,
        }
    )


def get_timezone_name(
    dt_or_tzinfo: _DtOrTzinfo = None,
    width: Literal['long', 'short'] = 'long',
    uncommon: bool = False,
    locale: Locale | str | None = None,
    zone_variant: Literal['generic', 'daylight', 'standard'] | None = None,
    return_zone: bool = False,
) -> str:
    r"""Return the localized display name for the given timezone. The timezone
    may be specified using a ``datetime`` or `tzinfo` object.

    >>> from datetime import time
    >>> dt = time(15, 30, tzinfo=get_timezone('America/Los_Angeles'))
    >>> get_timezone_name(dt, locale='en_US')  # doctest: +SKIP
    'Pacific Standard Time'
    >>> get_timezone_name(dt, locale='en_US', return_zone=True)
    'America/Los_Angeles'
    >>> get_timezone_name(dt, width='short', locale='en_US')  # doctest: +SKIP
    'PST'

    If this function gets passed only a `tzinfo` object and no concrete
    `datetime`,  the returned display name is independent of daylight savings
    time. This can be used for example for selecting timezones, or to set the
    time of events that recur across DST changes:

    >>> tz = get_timezone('America/Los_Angeles')
    >>> get_timezone_name(tz, locale='en_US')
    'Pacific Time'
    >>> get_timezone_name(tz, 'short', locale='en_US')
    'PT'

    If no localized display name for the timezone is available, and the timezone
    is associated with a country that uses only a single timezone, the name of
    that country is returned, formatted according to the locale:

    >>> tz = get_timezone('Europe/Berlin')
    >>> get_timezone_name(tz, locale='de_DE')
    'Mitteleuropäische Zeit'
    >>> get_timezone_name(tz, locale='pt_BR')
    'Horário da Europa Central'

    On the other hand, if the country uses multiple timezones, the city is also
    included in the representation:

    >>> tz = get_timezone('America/St_Johns')
    >>> get_timezone_name(tz, locale='de_DE')
    'Neufundland-Zeit'

    Note that short format is currently not supported for all timezones and
    all locales.  This is partially because not every timezone has a short
    code in every locale.  In that case it currently falls back to the long
    format.

    For more information see `LDML Appendix J: Time Zone Display Names
    <https://www.unicode.org/reports/tr35/#Time_Zone_Fallback>`_

    .. versionadded:: 0.9

    .. versionchanged:: 1.0
       Added `zone_variant` support.

    :param dt_or_tzinfo: the ``datetime`` or ``tzinfo`` object that determines
                         the timezone; if a ``tzinfo`` object is used, the
                         resulting display name will be generic, i.e.
                         independent of daylight savings time; if `None`, the
                         current date in UTC is assumed
    :param width: either "long" or "short"
    :param uncommon: deprecated and ignored
    :param zone_variant: defines the zone variation to return.  By default the
                           variation is defined from the datetime object
                           passed in.  If no datetime object is passed in, the
                           ``'generic'`` variation is assumed.  The following
                           values are valid: ``'generic'``, ``'daylight'`` and
                           ``'standard'``.
    :param locale: the `Locale` object, or a locale string. Defaults to the system time locale.
    :param return_zone: True or False. If true then function
                        returns long time zone ID
    """
    dt, tzinfo = _get_dt_and_tzinfo(dt_or_tzinfo)
    locale = Locale.parse(locale or LC_TIME)

    zone = _get_tz_name(dt_or_tzinfo)

    if zone_variant is None:
        if dt is None:
            zone_variant = 'generic'
        else:
            dst = tzinfo.dst(dt)
            zone_variant = "daylight" if dst else "standard"
    else:
        if zone_variant not in ('generic', 'standard', 'daylight'):
            raise ValueError('Invalid zone variation')

    # Get the canonical time-zone code
    zone = get_global('zone_aliases').get(zone, zone)
    if return_zone:
        return zone
    info = locale.time_zones.get(zone, {})
    # Try explicitly translated zone names first
    if width in info and zone_variant in info[width]:
        value = info[width][zone_variant]
        if value != NO_INHERITANCE_MARKER:
            return value

    metazone = get_global('meta_zones').get(zone)
    if metazone:
        metazone_info = locale.meta_zones.get(metazone, {})
        if width in metazone_info:
            name = metazone_info[width].get(zone_variant)
            if width == 'short' and name == NO_INHERITANCE_MARKER:
                # If the short form is marked no-inheritance,
                # try to fall back to the long name instead.
                name = metazone_info.get('long', {}).get(zone_variant)
            if name and name != NO_INHERITANCE_MARKER:
                return name

    # If we have a concrete datetime, we assume that the result can't be
    # independent of daylight savings time, so we return the GMT offset
    if dt is not None:
        return get_timezone_gmt(dt, width=width, locale=locale)

    return get_timezone_location(dt_or_tzinfo, locale=locale)


def format_date(
    date: datetime.date | None = None,
    format: _PredefinedTimeFormat | str = 'medium',
    locale: Locale | str | None = None,
) -> str:
    """Return a date formatted according to the given pattern.

    >>> from datetime import date
    >>> d = date(2007, 4, 1)
    >>> format_date(d, locale='en_US')
    'Apr 1, 2007'
    >>> format_date(d, format='full', locale='de_DE')
    'Sonntag, 1. April 2007'

    If you don't want to use the locale default formats, you can specify a
    custom date pattern:

    >>> format_date(d, "EEE, MMM d, ''yy", locale='en')
    "Sun, Apr 1, '07"

    :param date: the ``date`` or ``datetime`` object; if `None`, the current
                 date is used
    :param format: one of "full", "long", "medium", or "short", or a custom
                   date/time pattern
    :param locale: a `Locale` object or a locale identifier. Defaults to the system time locale.
    """
    if date is None:
        date = datetime.date.today()
    elif isinstance(date, datetime.datetime):
        date = date.date()

    locale = Locale.parse(locale or LC_TIME)
    if format in ('full', 'long', 'medium', 'short'):
        format = get_date_format(format, locale=locale)
    pattern = parse_pattern(format)
    return pattern.apply(date, locale)


def format_datetime(
    datetime: _Instant = None,
    format: _PredefinedTimeFormat | str = 'medium',
    tzinfo: datetime.tzinfo | None = None,
    locale: Locale | str | None = None,
) -> str:
    r"""Return a date formatted according to the given pattern.

    >>> from datetime import datetime
    >>> dt = datetime(2007, 4, 1, 15, 30)
    >>> format_datetime(dt, locale='en_US')
    'Apr 1, 2007, 3:30:00\u202fPM'

    For any pattern requiring the display of the timezone:

    >>> format_datetime(dt, 'full', tzinfo=get_timezone('Europe/Paris'),
    ...                 locale='fr_FR')
    'dimanche 1 avril 2007, 17:30:00 heure d’été d’Europe centrale'
    >>> format_datetime(dt, "yyyy.MM.dd G 'at' HH:mm:ss zzz",
    ...                 tzinfo=get_timezone('US/Eastern'), locale='en')
    '2007.04.01 AD at 11:30:00 EDT'

    :param datetime: the `datetime` object; if `None`, the current date and
                     time is used
    :param format: one of "full", "long", "medium", or "short", or a custom
                   date/time pattern
    :param tzinfo: the timezone to apply to the time for display
    :param locale: a `Locale` object or a locale identifier. Defaults to the system time locale.
    """
    datetime = _ensure_datetime_tzinfo(_get_datetime(datetime), tzinfo)

    locale = Locale.parse(locale or LC_TIME)
    if format in ('full', 'long', 'medium', 'short'):
        return (
            get_datetime_format(format, locale=locale)
            .replace("'", "")
            .replace('{0}', format_time(datetime, format, tzinfo=None, locale=locale))
            .replace('{1}', format_date(datetime, format, locale=locale))
        )
    else:
        return parse_pattern(format).apply(datetime, locale)


def format_time(
    time: datetime.time | datetime.datetime | float | None = None,
    format: _PredefinedTimeFormat | str = 'medium',
    tzinfo: datetime.tzinfo | None = None,
    locale: Locale | str | None = None,
) -> str:
    r"""Return a time formatted according to the given pattern.

    >>> from datetime import datetime, time
    >>> t = time(15, 30)
    >>> format_time(t, locale='en_US')
    '3:30:00\u202fPM'
    >>> format_time(t, format='short', locale='de_DE')
    '15:30'

    If you don't want to use the locale default formats, you can specify a
    custom time pattern:

    >>> format_time(t, "hh 'o''clock' a", locale='en')
    "03 o'clock PM"

    For any pattern requiring the display of the time-zone a
    timezone has to be specified explicitly:

    >>> t = datetime(2007, 4, 1, 15, 30)
    >>> tzinfo = get_timezone('Europe/Paris')
    >>> t = _localize(tzinfo, t)
    >>> format_time(t, format='full', tzinfo=tzinfo, locale='fr_FR')
    '15:30:00 heure d’été d’Europe centrale'
    >>> format_time(t, "hh 'o''clock' a, zzzz", tzinfo=get_timezone('US/Eastern'),
    ...             locale='en')
    "09 o'clock AM, Eastern Daylight Time"

    As that example shows, when this function gets passed a
    ``datetime.datetime`` value, the actual time in the formatted string is
    adjusted to the timezone specified by the `tzinfo` parameter. If the
    ``datetime`` is "naive" (i.e. it has no associated timezone information),
    it is assumed to be in UTC.

    These timezone calculations are **not** performed if the value is of type
    ``datetime.time``, as without date information there's no way to determine
    what a given time would translate to in a different timezone without
    information about whether daylight savings time is in effect or not. This
    means that time values are left as-is, and the value of the `tzinfo`
    parameter is only used to display the timezone name if needed:

    >>> t = time(15, 30)
    >>> format_time(t, format='full', tzinfo=get_timezone('Europe/Paris'),
    ...             locale='fr_FR')  # doctest: +SKIP
    '15:30:00 heure normale d\u2019Europe centrale'
    >>> format_time(t, format='full', tzinfo=get_timezone('US/Eastern'),
    ...             locale='en_US')  # doctest: +SKIP
    '3:30:00\u202fPM Eastern Standard Time'

    :param time: the ``time`` or ``datetime`` object; if `None`, the current
                 time in UTC is used
    :param format: one of "full", "long", "medium", or "short", or a custom
                   date/time pattern
    :param tzinfo: the time-zone to apply to the time for display
    :param locale: a `Locale` object or a locale identifier. Defaults to the system time locale.
    """

    # get reference date for if we need to find the right timezone variant
    # in the pattern
    ref_date = time.date() if isinstance(time, datetime.datetime) else None

    time = _get_time(time, tzinfo)

    locale = Locale.parse(locale or LC_TI