from __future__ import annotations

import datetime
import os
import typing as t

import pendulum

from pendulum.duration import Duration
from pendulum.parsing import _Interval
from pendulum.parsing import parse as base_parse
from pendulum.tz.timezone import UTC

if t.TYPE_CHECKING:
    from pendulum.date import Date
    from pendulum.datetime import DateTime
    from pendulum.interval import Interval
    from pendulum.time import Time


with_extensions = os.getenv("PENDULUM_EXTENSIONS", "1") == "1"

try:
    if not with_extensions:
        raise ImportError

    from pendulum._pendulum import Duration as RustDuration
except ImportError:
    RustDuration = None  # type: ignore[assignment,misc]


def parse(text: str, **options: t.Any) -> Date | Time | DateTime | Duration:
    options["now"] = options.get("now")
    return _parse(text, **options)


def _parse(
    text: str, **options: t.Any
) -> Date | DateTime | Time | Duration | Interval[DateTime]:
    if text == "now":
        return pendulum.now(tz=options.get("tz", UTC))

    value = base_parse(text, **options)

    if isinstance(value, datetime.datetime):
        return pendulum.datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tz=value.tzinfo or options.get("tz", UTC),
        )

    if isinstance(value, datetime.date):
        return pendulum.date(value.year, value.month, value.day)

    if isinstance(value, datetime.time):
        return pendulum.time(
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
        )

    if isinstance(value, _Interval):
        if value.duration is not None:
            duration = value.duration

            if value.start is not None:
                start = pendulum.instance(value.start, tz=options.get("tz", UTC))
                end = start.add(
                    years=duration.years,
                    months=duration.months,
                    weeks=duration.weeks,
                    days=duration.remaining_days,
                    hours=duration.hours,
                    minutes=duration.minutes,
                    seconds=duration.remaining_seconds,
                    microseconds=duration.microseconds,
                )
                return pendulum.interval(start, end)

            end = pendulum.instance(
                t.cast(datetime.datetime, value.end),
                tz=options.get("tz", UTC),
            )
            start = end.subtract(
                years=duration.years,
                months=duration.months,
                weeks=duration.weeks,
                days=duration.remaining_days,
                hours=duration.hours,
                minutes=duration.minutes,
                seconds=duration.remaining_seconds,
                microseconds=duration.microseconds,
            )
            return pendulum.interval(start, end)

        return pendulum.interval(
            pendulum.instance(
                t.cast(datetime.datetime, value.start),
                tz=options.get("tz", UTC),
            ),
            pendulum.instance(
                t.cast(datetime.datetime, value.end),
                tz=options.get("tz", UTC),
            ),
        )

    if isinstance(value, Duration):
        return value

    if RustDuration is not None and isinstance(value, RustDuration):
        return pendulum.duration(
            years=value.years,
            months=value.months,
            weeks=value.weeks,
            days=value.days,
            hours=value.hours,
            minutes=value.minutes,
            seconds=value.seconds,
            microseconds=value.microseconds,
        )

    raise NotImplementedError