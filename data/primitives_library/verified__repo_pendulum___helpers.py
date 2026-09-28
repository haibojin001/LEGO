from __future__ import annotations

import datetime
import math

from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

from pendulum.constants import DAY_OF_WEEK_TABLE
from pendulum.constants import DAYS_PER_L_YEAR
from pendulum.constants import DAYS_PER_MONTHS
from pendulum.constants import DAYS_PER_N_YEAR
from pendulum.constants import EPOCH_YEAR
from pendulum.constants import MONTHS_OFFSETS
from pendulum.constants import SECS_PER_4_YEARS
from pendulum.constants import SECS_PER_100_YEARS
from pendulum.constants import SECS_PER_400_YEARS
from pendulum.constants import SECS_PER_DAY
from pendulum.constants import SECS_PER_HOUR
from pendulum.constants import SECS_PER_MIN
from pendulum.constants import SECS_PER_YEAR
from pendulum.constants import TM_DECEMBER
from pendulum.constants import TM_JANUARY

if TYPE_CHECKING:
    import zoneinfo

    from pendulum.tz.timezone import Timezone


class PreciseDiff(NamedTuple):
    years: int
    months: int
    days: int
    hours: int
    minutes: int
    seconds: int
    microseconds: int
    total_days: int

    def __repr__(self) -> str:
        values = (
            ("years", self.years),
            ("months", self.months),
            ("days", self.days),
            ("hours", self.hours),
            ("minutes", self.minutes),
            ("seconds", self.seconds),
            ("microseconds", self.microseconds),
        )
        return " ".join(f"{value} {unit}" for unit, value in values)


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def is_long_year(year: int) -> bool:
    def week_anchor(value: int) -> int:
        return value + value // 4 - value // 100 + value // 400

    return week_anchor(year) % 7 == 4 or week_anchor(year - 1) % 7 == 3


def week_day(year: int, month: int, day: int) -> int:
    adjusted_year = year - 1 if month < 3 else year
    result = (
        adjusted_year
        + adjusted_year // 4
        - adjusted_year // 100
        + adjusted_year // 400
        + DAY_OF_WEEK_TABLE[month - 1]
        + day
    ) % 7
    return result if result else 7


def days_in_year(year: int) -> int:
    return DAYS_PER_L_YEAR if is_leap(year) else DAYS_PER_N_YEAR


def local_time(
    unix_time: int, utc_offset: int, microseconds: int
) -> tuple[int, int, int, int, int, int, int]:
    year = EPOCH_YEAR
    remaining = math.floor(unix_time)

    if remaining >= 0:
        remaining -= 10957 * SECS_PER_DAY
        year += 30
    else:
        remaining += (146097 - 10957) * SECS_PER_DAY
        year -= 370

    remaining += utc_offset

    cycles, remaining = divmod(remaining, SECS_PER_400_YEARS)
    year += cycles * 400

    leap_kind = 1
    century_seconds = SECS_PER_100_YEARS[leap_kind]
    while remaining >= century_seconds:
        remaining -= century_seconds
        year += 100
        leap_kind = 0
        century_seconds = SECS_PER_100_YEARS[leap_kind]

    block_seconds = SECS_PER_4_YEARS[leap_kind]
    while remaining >= block_seconds:
        remaining -= block_seconds
        year += 4
        leap_kind = 1
        block_seconds = SECS_PER_4_YEARS[leap_kind]

    current_year_seconds = SECS_PER_YEAR[leap_kind]
    while remaining >= current_year_seconds:
        remaining -= current_year_seconds
        year += 1
        leap_kind = 0
        current_year_seconds = SECS_PER_YEAR[leap_kind]

    month = TM_DECEMBER + 1
    day = remaining // SECS_PER_DAY + 1
    remaining %= SECS_PER_DAY

    while month != TM_JANUARY + 1:
        length = MONTHS_OFFSETS[leap_kind][month]
        if day > length:
            day -= length
            break
        month -= 1

    hour, remaining = divmod(remaining, SECS_PER_HOUR)
    minute, second = divmod(remaining, SECS_PER_MIN)

    return year, month, day, hour, minute, second, microseconds


def precise_diff(
    d1: datetime.datetime | datetime.date, d2: datetime.datetime | datetime.date
) -> PreciseDiff:
    multiplier = 1

    if d1 == d2:
        return PreciseDiff(0, 0, 0, 0, 0, 0, 0, 0)

    first_tz = d1.tzinfo if isinstance(d1, datetime.datetime) else None
    second_tz = d2.tzinfo if isinstance(d2, datetime.datetime) else None

    if (first_tz is None) != (second_tz is None):
        raise ValueError("Comparison between naive and aware datetimes is not supported")

    if d1 > d2:
        d1, d2 = d2, d1
        multiplier = -1

    total_days = _day_number(d2.year, d2.month, d2.day) - _day_number(
        d1.year, d1.month, d1.day
    )

    days = 0
    hours = 0
    minutes = 0
    seconds = 0
    microseconds = 0

    same_timezone = False
    if first_tz is not None and second_tz is not None:
        first_name = _get_tzinfo_name(first_tz)
        second_name = _get_tzinfo_name(second_tz)
        same_timezone = first_name is not None and first_name == second_name

    if isinstance(d2, datetime.datetime):
        if isinstance(d1, datetime.datetime):
            if not same_timezone or total_days == 0:
                first_offset = d1.utcoffset()
                second_offset = d2.utcoffset()

                if first_offset:
                    d1 = d1 - first_offset
                if second_offset:
                    d2 = d2 - second_offset

            hours = d2.hour - d1.hour
            minutes = d2.minute - d1.minute
            seconds = d2.second - d1.second
            microseconds = d2.microsecond - d1.microsecond
        else:
            hours = d2.hour
            minutes = d2.minute
            seconds = d2.second
            microseconds = d2.microsecond

        if microseconds < 0:
            microseconds += 1_000_000
            seconds -= 1
        if seconds < 0:
            seconds += 60
            minutes -= 1
        if minutes < 0:
            minutes += 60
            hours -= 1
        if hours < 0:
            hours += 24
            days -= 1

    years = d2.year - d1.year
    months = d2.month - d1.month
    days += d2.day - d1.day

    if days < 0:
        previous_month = d2.month - 1
        previous_year = d2.year
        if previous_month == 0:
            previous_month = 12
            previous_year -= 1

        preceding_length = DAYS_PER_MONTHS[int(is_leap(previous_year))][previous_month]
        ending_length = DAYS_PER_MONTHS[int(is_leap(d2.year))][d2.month]
        threshold = ending_length - preceding_length

        if days < threshold:
            if preceding_length < d1.day:
                days += d1.day
            else:
                days += preceding_length
        elif days == threshold:
            days = 0
            months += 1
        else:
            days += preceding_length

        months -= 1

    if months < 0:
        months += 12
        years -= 1

    return PreciseDiff(
        multiplier * years,
        multiplier * months,
        multiplier * days,
        multiplier * hours,
        multiplier * minutes,
        multiplier * seconds,
        multiplier * microseconds,
        multiplier * total_days,
    )


def _day_number(year: int, month: int, day: int) -> int:
    shifted_month = (month + 9) % 12
    shifted_year = year - shifted_month // 10
    return (
        365 * shifted_year
        + shifted_year // 4
        - shifted_year // 100
        + shifted_year // 400
        + (shifted_month * 306 + 5) // 10
        + day
        - 1
    )


def _get_tzinfo_name(tzinfo: datetime.tzinfo | None) -> str | None:
    if tzinfo is None:
        return None

    if hasattr(tzinfo, "key"):
        return cast("zoneinfo.ZoneInfo", tzinfo).key
    if hasattr(tzinfo, "name"):
        return cast("Timezone", tzinfo).name
    if hasattr(tzinfo, "zone"):
        return tzinfo.zone  # type: ignore[no-any-return]

    return None