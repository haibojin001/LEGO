from __future__ import annotations

import datetime
from enum import IntEnum
from typing import Any, Optional

from dateutil.rrule import WEEKLY, rrule


class WeekDay(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


MIN_ORDINAL = 1
MAX_ORDINAL = datetime.date.max.toordinal()
MAX_TIMESTAMP = 253402300799
MAX_TIMESTAMP_MS = MAX_TIMESTAMP * 1000
MAX_TIMESTAMP_US = MAX_TIMESTAMP_MS * 1000


def next_weekday(
    start_date: Optional[datetime.date], weekday: int
) -> datetime.datetime:
    if weekday not in range(7):
        raise ValueError("Weekday must be between 0 (Monday) and 6 (Sunday).")

    return rrule(
        freq=WEEKLY,
        dtstart=start_date,
        byweekday=weekday,
        count=1,
    )[0]


def is_timestamp(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False

    try:
        float(value)
    except ValueError:
        return False

    return True


def validate_ordinal(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Ordinal must be an integer (got type {type(value)}).")

    if value < MIN_ORDINAL or value > MAX_ORDINAL:
        raise ValueError(f"Ordinal {value} is out of range.")


def normalize_timestamp(timestamp: float) -> float:
    if timestamp <= MAX_TIMESTAMP:
        return timestamp

    if timestamp < MAX_TIMESTAMP_MS:
        return timestamp / 1000

    if timestamp < MAX_TIMESTAMP_US:
        return timestamp / 1_000_000

    raise ValueError(f"The specified timestamp {timestamp!r} is too large.")


def iso_to_gregorian(iso_year: int, iso_week: int, iso_day: int) -> datetime.date:
    if iso_week < 1 or iso_week > 53:
        raise ValueError("ISO Calendar week value must be between 1-53.")

    if iso_day < 1 or iso_day > 7:
        raise ValueError("ISO Calendar day value must be between 1-7")

    january_fourth = datetime.date(iso_year, 1, 4)
    first_monday = january_fourth - datetime.timedelta(
        days=january_fourth.isoweekday() - 1
    )
    return first_monday + datetime.timedelta(weeks=iso_week - 1, days=iso_day - 1)


def validate_bounds(bounds: str) -> None:
    if bounds not in ("()", "(]", "[)", "[]"):
        raise ValueError(
            "Invalid bounds. Please select between '()', '(]', '[)', or '[]'."
        )


__all__ = [
    "next_weekday",
    "is_timestamp",
    "validate_ordinal",
    "iso_to_gregorian",
]