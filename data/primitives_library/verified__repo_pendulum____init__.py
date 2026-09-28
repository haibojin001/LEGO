from __future__ import annotations

import contextlib
import copy
import os
import re

from datetime import date
from datetime import datetime
from datetime import time
from typing import Any
from typing import cast

from dateutil import parser

from pendulum.parsing.exceptions import ParserError


with_extensions = os.getenv("PENDULUM_EXTENSIONS", "1") == "1"

try:
    if not with_extensions:
        raise ImportError

    from pendulum._pendulum import Duration
    from pendulum._pendulum import parse_iso8601
except ImportError:
    from pendulum.duration import Duration
    from pendulum.parsing.iso8601 import parse_iso8601


COMMON = re.compile(
    r"""
    ^
    (?P<date>
        (?P<classic>
            (?P<year>\d{4})
            (?P<monthday>
                (?P<monthsep>[/:])?(?P<month>\d{2})
                ((?P<daysep>[/:])?(?P<day>\d{2}))
            )?
        )
    )?
    (?P<time>
        (?P<timesep>\ )?
        (?P<hour>\d{1,2}):(?P<minute>\d{1,2})?(?::(?P<second>\d{1,2}))?
        (?P<subsecondsection>
            (?:[.|,])
            (?P<subsecond>\d{1,9})
        )?
    )?
    $
    """,
    re.VERBOSE,
)

DEFAULT_OPTIONS = {
    "day_first": False,
    "year_first": True,
    "strict": True,
    "exact": False,
    "now": None,
}


def parse(text: str, **options: Any) -> datetime | date | time | _Interval | Duration:
    settings: dict[str, Any] = copy.copy(DEFAULT_OPTIONS)
    settings.update(options)
    return _normalize(_parse(text, **settings), **settings)


def _normalize(
    parsed: datetime | date | time | _Interval | Duration, **options: Any
) -> datetime | date | time | _Interval | Duration:
    if options.get("exact"):
        return parsed

    if isinstance(parsed, time):
        reference = cast("datetime | None", options["now"]) or datetime.now()
        return datetime(
            reference.year,
            reference.month,
            reference.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
            parsed.microsecond,
        )

    if isinstance(parsed, date) and not isinstance(parsed, datetime):
        return datetime(parsed.year, parsed.month, parsed.day)

    return parsed


def _parse(text: str, **options: Any) -> datetime | date | time | _Interval | Duration:
    with contextlib.suppress(ValueError):
        return parse_iso8601(text)

    with contextlib.suppress(ValueError):
        return _parse_iso8601_interval(text)

    with contextlib.suppress(ParserError):
        return _parse_common(text, **options)

    if options.get("strict", True):
        raise ParserError(f"Unable to parse string [{text}]")

    try:
        return parser.parse(
            text,
            dayfirst=options["day_first"],
            yearfirst=options["year_first"],
        )
    except ValueError:
        raise ParserError(f"Invalid date string: {text}")


def _parse_common(text: str, **options: Any) -> datetime | date | time:
    match = COMMON.fullmatch(text)

    date_present = False
    year = 0
    month = 1
    day = 1

    if match is None or not (match.group("date") or match.group("time")):
        raise ParserError("Invalid datetime string")

    if match.group("date"):
        date_present = True
        year = int(match.group("year"))

        if match.group("monthday"):
            if options["day_first"]:
                month = int(match.group("day"))
                day = int(match.group("month"))
            else:
                month = int(match.group("month"))
                day = int(match.group("day"))

    if not match.group("time"):
        return date(year, month, day)

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second")) if match.group("second") else 0

    microsecond = 0
    if match.group("subsecondsection"):
        fraction = match.group("subsecond")[:6]
        microsecond = int(f"{fraction:0<6}")

    if date_present:
        return datetime(year, month, day, hour, minute, second, microsecond)

    return time(hour, minute, second, microsecond)


class _Interval:
    def __init__(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        duration: Duration | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.duration = duration


def _parse_iso8601_interval(text: str) -> _Interval:
    if "/" not in text:
        raise ParserError("Invalid interval")

    first, last = text.split("/")

    if not first or not last:
        raise ParserError("Invalid interval.")

    start = None
    end = None
    duration = None

    if first[:1] == "P":
        duration = parse_iso8601(first)
        end = parse_iso8601(last)
    elif last[:1] == "P":
        start = parse_iso8601(first)
        duration = parse_iso8601(last)
    else:
        start = parse_iso8601(first)
        end = parse_iso8601(last)

    return _Interval(
        cast("datetime", start),
        cast("datetime", end),
        cast("Duration", duration),
    )


__all__ = ["parse", "parse_iso8601"]