from __future__ import annotations

import datetime
import re
from typing import cast

from pendulum.constants import HOURS_PER_DAY
from pendulum.constants import MINUTES_PER_HOUR
from pendulum.constants import MONTHS_OFFSETS
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.duration import Duration
from pendulum.helpers import days_in_year
from pendulum.helpers import is_leap
from pendulum.helpers import is_long_year
from pendulum.helpers import week_day
from pendulum.parsing.exceptions import ParserError
from pendulum.tz.timezone import UTC
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


ISO8601_DT = re.compile(
    r"""
    ^
    (?P<date>
        (?P<classic>
            (?P<year>\d{4})
            (?P<monthday>
                (?P<monthsep>-)?(?P<month>\d{2})
                ((?P<daysep>-)?(?P<day>\d{1,2}))?
            )?
        )
        |
        (?P<isocalendar>
            (?P<isoyear>\d{4})
            (?P<weeksep>-)?
            W
            (?P<isoweek>\d{2})
            (?P<weekdaysep>-)?
            (?P<isoweekday>\d)?
        )
    )?
    (?P<time>
        (?P<timesep>[T\ ])?
        (?P<hour>\d{1,2})(?P<minsep>:)?(?P<minute>\d{1,2})?(?P<secsep>:)?(?P<second>\d{1,2})?
        (?P<subsecondsection>
            (?:[.,])
            (?P<subsecond>\d{1,9})
        )?
        (?P<tz>
            (?:[-+])\d{2}:?(?:\d{2})?|Z
        )?
    )?
    $
    """,
    re.VERBOSE,
)

ISO8601_DURATION = re.compile(
    r"""
    ^P
    (?P<w>
        (?P<weeks>\d+(?:[.,]\d+)?W)
    )?
    (?P<ymd>
        (?P<years>\d+(?:[.,]\d+)?Y)?
        (?P<months>\d+(?:[.,]\d+)?M)?
        (?P<days>\d+(?:[.,]\d+)?D)?
    )?
    (?P<hms>
        (?P<timesep>T)
        (?P<hours>\d+(?:[.,]\d+)?H)?
        (?P<minutes>\d+(?:[.,]\d+)?M)?
        (?P<seconds>\d+(?:[.,]\d+)?S)?
    )?
    $
    """,
    re.VERBOSE,
)


def _get_iso_8601_week(
    year: str | int, week: str | int, weekday: str | int | None = None
) -> dict[str, int]:
    iso_year = int(year)
    iso_week = int(week)
    iso_weekday = int(weekday) if weekday is not None else 1

    if iso_week < 1 or iso_week > 53:
        raise ParserError("Invalid ISO 8601 week")

    if iso_weekday < 1 or iso_weekday > 7:
        raise ParserError("Invalid ISO 8601 weekday")

    if iso_week == 53 and not is_long_year(iso_year):
        raise ParserError("Invalid ISO 8601 week")

    first_weekday = week_day(iso_year, 1, 1)
    ordinal = (iso_week - 1) * 7 + iso_weekday

    if first_weekday <= 4:
        ordinal -= first_weekday - 1
    else:
        ordinal += 8 - first_weekday

    result_year = iso_year
    if ordinal < 1:
        result_year -= 1
        ordinal += days_in_year(result_year)
    elif ordinal > days_in_year(result_year):
        ordinal -= days_in_year(result_year)
        result_year += 1

    offsets = MONTHS_OFFSETS[is_leap(result_year)]
    month = 1
    while month <= 12 and ordinal > offsets[month]:
        month += 1

    return {
        "year": result_year,
        "month": month,
        "day": ordinal - offsets[month - 1],
    }


def parse_iso8601(
    text: str,
) -> datetime.datetime | datetime.date | datetime.time | Duration:
    parsed_duration = _parse_iso8601_duration(text)
    if parsed_duration is not None:
        return parsed_duration

    match = ISO8601_DT.fullmatch(text)
    if match is None:
        raise ParserError("Invalid ISO 8601 string")

    ambiguous_date = False
    has_date = False
    has_time_only = False

    year = 0
    month = 1
    day = 1
    minute = 0
    second = 0
    microsecond = 0
    tzinfo: FixedTimezone | Timezone | None = None

    if match.group("date"):
        has_date = True

        if match.group("isocalendar"):
            if (
                match.group("weeksep")
                and not match.group("weekdaysep")
                and match.group("isoweekday")
            ):
                raise ParserError(f"Invalid date string: {text}")

            if not match.group("weeksep") and match.group("weekdaysep"):
                raise ParserError(f"Invalid date string: {text}")

            try:
                iso_date = _get_iso_8601_week(
                    cast(str, match.group("isoyear")),
                    cast(str, match.group("isoweek")),
                    match.group("isoweekday"),
                )
            except ParserError:
                raise
            except ValueError:
                raise ParserError(f"Invalid date string: {text}")

            year = iso_date["year"]
            month = iso_date["month"]
            day = iso_date["day"]
        else:
            year = int(cast(str, match.group("year")))

            if not match.group("monthday"):
                month = 1
                day = 1
            elif match.group("month") and match.group("day"):
                if not match.group("daysep") and len(cast(str, match.group("day"))) == 1:
                    ordinal = int(
                        cast(str, match.group("month")) + cast(str, match.group("day"))
                    )
                    offsets = MONTHS_OFFSETS[is_leap(year)]

                    if ordinal > offsets[13]:
                        raise ParserError("Ordinal day is out of range")

                    for index in range(1, 14):
                        if ordinal <= offsets[index]:
                            month = index - 1
                            day = ordinal - offsets[index - 1]
                            break
                else:
                    month = int(cast(str, match.group("month")))
                    day = int(cast(str, match.group("day")))
            else:
                if not match.group("monthsep"):
                    ambiguous_date = True

                month = int(cast(str, match.group("month")))
                day = 1

    if not match.group("time"):
        if ambiguous_date:
            value = f"{year}{month:0>2}"
            return datetime.time(int(value[:2]), int(value[2:4]), int(value[4:]))

        return datetime.date(year, month, day)

    if ambiguous_date:
        raise ParserError(f"Invalid date string: {text}")

    if has_date and not match.group("timesep"):
        raise ParserError(f"Invalid date string: {text}")

    if not has_date:
        has_time_only = True

    hour = int(cast(str, match.group("hour")))
    minute_separator = match.group("minsep")

    if match.group("minute"):
        minute = int(cast(str, match.group("minute")))
    elif minute_separator:
        raise ParserError("Invalid ISO 8601 time part")

    second_separator = match.group("secsep")
    if second_separator and not minute_separator and match.group("minute"):
        raise ParserError("Invalid ISO 8601 time part")

    if match.group("second"):
        if not second_separator and minute_separator:
            raise ParserError("Invalid ISO 8601 time part")

        second = int(cast(str, match.group("second")))
    elif second_separator:
        raise ParserError("Invalid ISO 8601 time part")

    if match.group("subsecondsection"):
        fraction = cast(str, match.group("subsecond"))[:6]
        microsecond = int(f"{fraction:0<6}")

    timezone = match.group("tz")
    if timezone:
        if timezone == "Z":
            tzinfo = UTC
        else:
            negative = timezone.startswith("-")
            offset_text = timezone[1:]

            if ":" in offset_text:
                offset_hour, offset_minute = offset_text.split(":")
            else:
                if len(offset_text) == 2:
                    offset_text = f"{offset_text}00"

                offset_hour = offset_text[:2]
                offset_minute = offset_text[2:4]

            offset = (int(offset_hour) * MINUTES_PER_HOUR + int(offset_minute)) * SECONDS_PER_MINUTE
            if negative:
                offset = -offset

            tzinfo = FixedTimezone(offset)

    if has_time_only:
        return datetime.time(hour, minute, second, microsecond, tzinfo=tzinfo)

    return datetime.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=tzinfo,
    )


def _parse_iso8601_duration(text: str, **options: str) -> Duration | None:
    match = ISO8601_DURATION.fullmatch(text)
    if match is None or (
        not match.group("w") and not match.group("ymd") and not match.group("hms")
    ):
        return None

    if match.group("w") and (match.group("ymd") or match.group("hms")):
        raise ParserError("Invalid duration string")

    def number(value: str | None, suffix: str) -> int | float:
        if not value:
            return 0

        raw = value[: -len(suffix)].replace(",", ".")
        if "." in raw:
            return float(raw)

        return int(raw)

    years_value = number(match.group("years"), "Y")
    months_value = number(match.group("months"), "M")
    weeks_value = number(match.group("weeks"), "W")
    days_value = number(match.group("days"), "D")
    hours_value = number(match.group("hours"), "H")
    minutes_value = number(match.group("minutes"), "M")
    seconds_value = number(match.group("seconds"), "S")

    if isinstance(years_value, float) and not years_value.is_integer():
        raise ParserError("Invalid duration string")
    if isinstance(months_value, float) and not months_value.is_integer():
        raise ParserError("Invalid duration string")

    years = int(years_value)
    months = int(months_value)

    whole_weeks = int(weeks_value)
    fractional_weeks = float(weeks_value) - whole_weeks

    total_microseconds = 0.0
    total_microseconds += fractional_weeks * 7 * HOURS_PER_DAY * MINUTES_PER_HOUR * SECONDS_PER_MINUTE * 1_000_000
    total_microseconds += float(days_value) * HOURS_PER_DAY * MINUTES_PER_HOUR * SECONDS_PER_MINUTE * 1_000_000
    total_microseconds += float(hours_value) * MINUTES_PER_HOUR * SECONDS_PER_MINUTE * 1_000_000
    total_microseconds += float(minutes_value) * SECONDS_PER_MINUTE * 1_000_000
    total_microseconds += float(seconds_value) * 1_000_000

    total_microseconds_int = int(round(total_microseconds))
    day_microseconds = HOURS_PER_DAY * MINUTES_PER_HOUR * SECONDS_PER_MINUTE * 1_000_000
    hour_microseconds = MINUTES_PER_HOUR * SECONDS_PER_MINUTE * 1_000_000
    minute_microseconds = SECONDS_PER_MINUTE * 1_000_000

    days, remainder = divmod(total_microseconds_int, day_microseconds)
    hours, remainder = divmod(remainder, hour_microseconds)
    minutes, remainder = divmod(remainder, minute_microseconds)
    seconds, microseconds = divmod(remainder, 1_000_000)

    return Duration(
        years=years,
        months=months,
        weeks=whole_weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
    )