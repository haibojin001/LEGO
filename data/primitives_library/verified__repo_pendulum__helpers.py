from __future__ import annotations

import os
import struct
from datetime import date
from datetime import datetime
from datetime import timedelta
from functools import cache
from math import copysign
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import overload

import pendulum
import pendulum.exceptions as _exceptions

from pendulum.constants import DAYS_PER_MONTHS
from pendulum.day import WeekDay

if not hasattr(_exceptions, "InvalidLocale"):

    class InvalidLocale(_exceptions.PendulumException):
        pass

    _exceptions.InvalidLocale = InvalidLocale

if TYPE_CHECKING:
    from pendulum.duration import Duration
    from pendulum.locales.locale import Locale
    from pendulum.formatting.difference_formatter import DifferenceFormatter

with_extensions = os.getenv("PENDULUM_EXTENSIONS", "1") == "1"

_DT = TypeVar("_DT", bound=datetime)
_D = TypeVar("_D", bound=date)

try:
    if not with_extensions or struct.calcsize("P") == 4:
        raise ImportError

    from pendulum._pendulum import PreciseDiff
    from pendulum._pendulum import days_in_year
    from pendulum._pendulum import is_leap
    from pendulum._pendulum import is_long_year
    from pendulum._pendulum import local_time
    from pendulum._pendulum import precise_diff
    from pendulum._pendulum import week_day
except ImportError:
    from pendulum._helpers import PreciseDiff
    from pendulum._helpers import days_in_year
    from pendulum._helpers import is_leap
    from pendulum._helpers import is_long_year
    from pendulum._helpers import local_time
    from pendulum._helpers import precise_diff
    from pendulum._helpers import week_day

difference_formatter: DifferenceFormatter


def _sign(value: float) -> int:
    return int(copysign(1, value))


@overload
def add_duration(
    dt: _DT,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
    microseconds: int = 0,
) -> _DT: ...


@overload
def add_duration(
    dt: _D,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
) -> _D: ...


def add_duration(
    dt: date | datetime,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
    microseconds: int = 0,
) -> date | datetime:
    days += weeks * 7

    if (
        isinstance(dt, date)
        and not isinstance(dt, datetime)
        and any((hours, minutes, seconds, microseconds))
    ):
        raise RuntimeError("Time elements cannot be added to a date instance.")

    if abs(microseconds) > 999999:
        direction = _sign(microseconds)
        quotient, remainder = divmod(microseconds * direction, 1000000)
        microseconds = remainder * direction
        seconds += quotient * direction

    if abs(seconds) > 59:
        direction = _sign(seconds)
        quotient, remainder = divmod(seconds * direction, 60)
        seconds = remainder * direction
        minutes += quotient * direction

    if abs(minutes) > 59:
        direction = _sign(minutes)
        quotient, remainder = divmod(minutes * direction, 60)
        minutes = remainder * direction
        hours += quotient * direction

    if abs(hours) > 23:
        direction = _sign(hours)
        quotient, remainder = divmod(hours * direction, 24)
        hours = remainder * direction
        days += quotient * direction

    if abs(months) > 11:
        direction = _sign(months)
        quotient, remainder = divmod(months * direction, 12)
        months = remainder * direction
        years += quotient * direction

    target_year = dt.year + years
    target_month = dt.month

    if months:
        target_month += months
        if target_month > 12:
            target_year += 1
            target_month -= 12
        elif target_month < 1:
            target_year -= 1
            target_month += 12

    target_day = min(
        DAYS_PER_MONTHS[int(is_leap(target_year))][target_month],
        dt.day,
    )

    adjusted = dt.replace(
        year=target_year,
        month=target_month,
        day=target_day,
    )

    return adjusted + timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
    )


def locale(name: str) -> Locale:
    from pendulum.locales import load_locale

    return load_locale(name)


def set_locale(name: str) -> None:
    locale(name)
    pendulum._LOCALE = name


def get_locale() -> str:
    return pendulum._LOCALE


def week_starts_at(wday: WeekDay) -> None:
    if wday < WeekDay.MONDAY or wday > WeekDay.SUNDAY:
        raise ValueError("Invalid day of week")

    pendulum._WEEK_STARTS_AT = wday


def week_ends_at(wday: WeekDay) -> None:
    if wday < WeekDay.MONDAY or wday > WeekDay.SUNDAY:
        raise ValueError("Invalid day of week")

    pendulum._WEEK_ENDS_AT = wday


@cache
def _difference_formatter() -> DifferenceFormatter:
    from pendulum.formatting.difference_formatter import DifferenceFormatter

    return DifferenceFormatter()


def format_diff(
    diff: Duration,
    is_now: bool = True,
    absolute: bool = False,
    locale: str | None = None,
) -> str:
    if locale is None:
        locale = get_locale()

    return _difference_formatter().format(diff, is_now, absolute, locale)


def __getattr__(name: str) -> Any:
    if name == "difference_formatter":
        return _difference_formatter()

    raise AttributeError(name)


__all__ = [
    "PreciseDiff",
    "add_duration",
    "days_in_year",
    "format_diff",
    "get_locale",
    "is_leap",
    "is_long_year",
    "local_time",
    "locale",
    "precise_diff",
    "set_locale",
    "week_day",
    "week_ends_at",
    "week_starts_at",
]