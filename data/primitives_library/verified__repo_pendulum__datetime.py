from __future__ import annotations

import calendar
import datetime
import traceback

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import ATOM
from pendulum.constants import COOKIE
from pendulum.constants import MINUTES_PER_HOUR
from pendulum.constants import MONTHS_PER_YEAR
from pendulum.constants import RFC822
from pendulum.constants import RFC850
from pendulum.constants import RFC1036
from pendulum.constants import RFC1123
from pendulum.constants import RFC2822
from pendulum.constants import RSS
from pendulum.constants import SECONDS_PER_DAY
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.constants import W3C
from pendulum.constants import YEARS_PER_CENTURY
from pendulum.constants import YEARS_PER_DECADE
from pendulum.date import Date
from pendulum.day import WeekDay
from pendulum.exceptions import PendulumException
from pendulum.interval import Interval
from pendulum.tz import UTC
from pendulum.tz import local_timezone
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from typing_extensions import Self
    from typing_extensions import SupportsIndex


class DateTime(datetime.datetime, Date):
    EPOCH: ClassVar["DateTime"]
    min: ClassVar["DateTime"]
    max: ClassVar["DateTime"]

    _FORMATS: ClassVar[dict[str, str | Callable[[datetime.datetime], str]]] = {
        "atom": ATOM,
        "cookie": COOKIE,
        "iso8601": lambda dt: dt.isoformat("T"),
        "rfc822": RFC822,
        "rfc850": RFC850,
        "rfc1036": RFC1036,
        "rfc1123": RFC1123,
        "rfc2822": RFC2822,
        "rfc3339": lambda dt: dt.isoformat("T"),
        "rss": RSS,
        "w3c": W3C,
    }

    _MODIFIERS_VALID_UNITS: ClassVar[list[str]] = [
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
    ]

    _EPOCH: datetime.datetime = datetime.datetime(1970, 1, 1, tzinfo=UTC)

    @classmethod
    def create(
        cls,
        year: SupportsIndex,
        month: SupportsIndex,
        day: SupportsIndex,
        hour: SupportsIndex = 0,
        minute: SupportsIndex = 0,
        second: SupportsIndex = 0,
        microsecond: SupportsIndex = 0,
        tz: str | float | Timezone | FixedTimezone | None | datetime.tzinfo = UTC,
        fold: int = 1,
        raise_on_unknown_times: bool = False,
    ) -> Self:
        if tz is not None:
            tz = pendulum._safe_timezone(tz)

        dt = datetime.datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            fold=fold,
        )

        if tz is not None:
            if hasattr(tz, "convert"):
                dt = tz.convert(dt, raise_on_unknown_times=raise_on_unknown_times)
            else:
                dt = dt.replace(tzinfo=tz)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=dt.tzinfo,
            fold=dt.fold,
        )

    @classmethod
    def instance(
        cls,
        dt: datetime.datetime,
        tz: str | Timezone | FixedTimezone | datetime.tzinfo | None = UTC,
    ) -> Self:
        tz = dt.tzinfo or tz

        if tz is not None:
            tz = pendulum._safe_timezone(tz, dt=dt)

        return cls.create(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tz=tz,
            fold=dt.fold,
        )

    @overload
    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> Self: ...

    @overload
    @classmethod
    def now(cls, tz: str | Timezone | FixedTimezone | None = None) -> Self: ...

    @classmethod
    def now(
        cls,
        tz: str | Timezone | FixedTimezone | datetime.tzinfo | None = None,
    ) -> Self:
        if tz is None or tz == "local":
            dt = datetime.datetime.now(local_timezone())
        elif tz is UTC or tz == "UTC":
            dt = datetime.datetime.now(UTC)
        else:
            dt = datetime.datetime.now(UTC)
            tz = pendulum._safe_timezone(tz)
            dt = dt.astimezone(tz)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=dt.tzinfo,
            fold=dt.fold,
        )

    @classmethod
    def utcnow(cls) -> Self:
        return cls.now(UTC)

    @classmethod
    def today(cls) -> Self:
        return cls.now()

    @classmethod
    def strptime(cls, time: str, fmt: str) -> Self:
        return cls.instance(datetime.datetime.strptime(time, fmt))

    @classmethod
    def fromtimestamp(
        cls,
        timestamp: float,
        tz: datetime.tzinfo | None = None,
    ) -> Self:
        return cls.instance(datetime.datetime.fromtimestamp(timestamp, tz), tz=tz)

    @classmethod
    def utcfromtimestamp(cls, timestamp: float) -> Self:
        return cls.instance(datetime.datetime.fromtimestamp(timestamp, UTC), tz=UTC)

    @classmethod
    def combine(
        cls,
        date: datetime.date,
        time: datetime.time,
        tzinfo: datetime.tzinfo | None = None,
    ) -> Self:
        if tzinfo is None:
            tzinfo = time.tzinfo
        return cls.instance(datetime.datetime.combine(date, time, tzinfo=tzinfo), tz=tzinfo)

    @classmethod
    def fromordinal(cls, n: int) -> Self:
        return cls.instance(datetime.datetime.fromordinal(n), tz=None)

    @classmethod
    def fromisoformat(cls, date_string: str) -> Self:
        return cls.instance(datetime.datetime.fromisoformat(date_string), tz=None)

    def set(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        hour: int | None = None,
        minute: int | None = None,
        second: int | None = None,
        microsecond: int | None = None,
        tz: str | float | Timezone | FixedTimezone | datetime.tzinfo | None = None,
    ) -> Self:
        return self.__class__.create(
            self.year if year is None else year,
            self.month if month is None else month,
            self.day if day is None else day,
            self.hour if hour is None else hour,
            self.minute if minute is None else minute,
            self.second if second is None else second,
            self.microsecond if microsecond is None else microsecond,
            tz=self.tz if tz is None else tz,
            fold=self.fold,
        )

    @property
    def float_timestamp(self) -> float:
        return self.timestamp()

    @property
    def int_timestamp(self) -> int:
        value = datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            tzinfo=self.tzinfo,
            fold=self.fold,
        )
        elapsed = value - self._EPOCH
        return elapsed.days * SECONDS_PER_DAY + elapsed.seconds

    @property
    def offset(self) -> int | None:
        return self.get_offset()

    @property
    def offset_hours(self) -> float | None:
        offset = self.get_offset()
        if offset is None:
            return None
        return offset / SECONDS_PER_MINUTE / MINUTES_PER_HOUR

    @property
    def timezone(self) -> Timezone | FixedTimezone | None:
        if isinstance(self.tzinfo, (Timezone, FixedTimezone)):
            return self.tzinfo
        return None

    @property
    def tz(self) -> Timezone | FixedTimezone | None:
        return self.timezone

    @property
    def timezone_name(self) -> str | None:
        timezone = self.timezone
        if timezone is not None:
            return timezone.name
        return self.tzname()

    @property
    def age(self) -> int:
        return self.date().diff(self.now(self.tz).date(), abs=False).in_years()

    def is_local(self) -> bool:
        return self.offset == self.in_timezone(local_timezone()).offset

    def is_utc(self) -> bool:
        return self.offset == 0

    def is_dst(self) -> bool:
        dst = self.dst()
        return dst is not None and dst != datetime.timedelta()

    def get_offset(self) -> int | None:
        value = self.utcoffset()
        return None if value is None else int(value.total_seconds())

    def date(self) -> Date:
        return Date(self.year, self.month, self.day)

    def time(self) -> Any:
        try:
            from pendulum.time import Time

            return Time(self.hour, self.minute, self.second, self.microsecond)
        except (ImportError, AttributeError):
            return datetime.time(
                self.hour,
                self.minute,
                self.second,
                self.microsecond,
                fold=self.fold,
            )

    def naive(self) -> Self:
        return self.__class__(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            fold=self.fold,
        )

    def on(self, year: int, month: int, day: int) -> Self:
        return self.set(year=int(year), month=int(month), day=int(day))

    def at(
        self,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> Self:
        return self.set(
            hour=int(hour),
            minute=int(minute),
            second=int(second),
            microsecond=int(microsecond),
        )

    def in_timezone(
        self,
        tz: str | float | Timezone | FixedTimezone | datetime.tzinfo,
    ) -> Self:
        timezone = pendulum._safe_timezone(tz)
        dt = datetime.datetime.astimezone(self, timezone)
        return self.__class__(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=dt.tzinfo,
            fold=dt.fold,
        )

    def in_tz(
        self,
        tz: str | float | Timezone | FixedTimezone | datetime.tzinfo,
    ) -> Self:
        return self.in_timezone(tz)

    def in_utc(self) -> Self:
        return self.in_timezone(UTC)

    def _with_timedelta(self, delta: datetime.timedelta) -> Self:
        dt = datetime.datetime.__add__(self, delta)
        return self.__class__.instance(dt, tz=dt.tzinfo)

    def add(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Self:
        years = int(years)
        months = int(months)

        if years or months:
            month_index = self.month - 1 + months + years * MONTHS_PER_YEAR
            year = self.year + month_index // MONTHS_PER_YEAR
            month = month_index % MONTHS_PER_YEAR + 1
            day = min(self.day, calendar.monthrange(year, month)[1])
            result = self.__class__.create(
                year,
                month,
                day,
                self.hour,
                self.minute,
                self.second,
                self.microsecond,
                tz=self.tzinfo,
                fold=self.fold,
            )
        else:
            result = self

        if weeks or days or hours or minutes or seconds or microseconds:
            result = result._with_timedelta(
                datetime.timedelta(
                    weeks=weeks,
                    days=days,
                    hours=hours,
                    minutes=minutes,
                    seconds=seconds,
                    microseconds=microseconds,
                )
            )

        return result

    def subtract(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Self:
        return self.add(
            years=-years,
            months=-months,
            weeks=-weeks,
            days=-days,
            hours=-hours,
            minutes=-minutes,
            seconds=-seconds,
            microseconds=-microseconds,
        )

    def start_of(self, unit: str) -> Self:
        unit = unit.lower().rstrip("s")

        if unit == "year":
            return self.set(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == "quarter":
            month = ((self.month - 1) // 3) * 3 + 1
            return self.set(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == "month":
            return self.set(day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == "week":
            return self.subtract(days=self.weekday()).start_of("day")
        if unit == "day":
            return self.set(hour=0, minute=0, second=0, microsecond=0)
        if unit == "hour":
            return self.set(minute=0, second=0, microsecond=0)
        if unit == "minute":
            return self.set(second=0, microsecond=0)
        if unit == "second":
            return self.set(microsecond=0)

        raise ValueError(f"Invalid unit '{unit}' for start_of().")

    def end_of(self, unit: str) -> Self:
        unit = unit.lower().rstrip("s")

        if unit == "year":
            return self.set(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        if unit == "quarter":
            month = ((self.month - 1) // 3) * 3 + 3
            return self.set(
                month=month,
                day=calendar.monthrange(self.year, month)[1],
                hour=23,
                minute=59,
                second=59,
                microsecond=999999,
            )
        if unit == "month":
            return self.set(
                day=calendar.monthrange(self.year, self.month)[1],
                hour=23,
                minute=59,
                second=59,
                microsecond=999999,
            )
        if unit == "week":
            return self.start_of("week").add(days=6).end_of("day")
        if unit == "day":
            return self.set(hour=23, minute=59, second=59, microsecond=999999)
        if unit == "hour":
            return self.set(minute=59, second=59, microsecond=999999)
        if unit == "minute":
            return self.set(second=59, microsecond=999999)
        if unit == "second":
            return self.set(microsecond=999999)

        raise ValueError(f"Invalid unit '{unit}' for end_of().")

    def next(self, weekday: WeekDay | int | None = None) -> Self:
        if weekday is None:
            return self.add(days=1)

        target = int(weekday)
        days = (target - self.weekday()) % 7
        if days == 0:
            days = 7
        return self.add(days=days)

    def previous(self, weekday: WeekDay | int | None = None) -> Self:
        if weekday is None:
            return self.subtract(days=1)

        target = int(weekday)
        days = (self.weekday() - target) % 7
        if days == 0:
            days = 7
        return self.subtract(days=days)

    def diff(
        self,
        dt: datetime.datetime | None = None,
        abs: bool = True,
    ) -> Interval:
        if dt is None:
            dt = self.now(self.tz)
        if not isinstance(dt, DateTime):
            dt = self.__class__.instance(dt, tz=dt.tzinfo)
        return Interval(self, dt, absolute=abs)

    def diff_for_humans(
        self,
        other: datetime.datetime | None = None,
        absolute: bool = False,
        locale: str | None = None,
    ) -> str:
        interval = self.diff(other, abs=absolute)
        if hasattr(interval, "in_words"):
            try:
                return interval.in_words(locale=locale)
            except TypeError:
                return interval.in_words()
        return str(interval)

    def format(self, fmt: str, locale: str | None = None) -> str:
        try:
            from pendulum.formatting.formatter import Formatter

            return Formatter().format(self, fmt, locale=locale)
        except Exception:
            return self.strftime(fmt)

    def _format(self, fmt: str) -> str:
        value = self._FORMATS[fmt]
        if callable(value):
            return value(self)
        return self.strftime(value)

    def to_atom_string(self) -> str:
        return self._format("atom")

    def to_cookie_string(self) -> str:
        return self._format("cookie")

    def to_iso8601_string(self) -> str:
        return self._format("iso8601")

    def to_rfc822_string(self) -> str:
        return self._format("rfc822")

    def to_rfc850_string(self) -> str:
        return self._format("rfc850")

    def to_rfc1036_string(self) -> str:
        return self._format("rfc1036")

    def to_rfc1123_string(self) -> str:
        return self._format("rfc1123")

    def to_rfc2822_string(self) -> str:
        return self._format("rfc2822")

    def to_rfc3339_string(self) -> str:
        return self._format("rfc3339")

    def to_rss_string(self) -> str:
        return self._format("rss")

    def to_w3c_string(self) -> str:
        return self._format("w3c")

    def to_datetime_string(self) -> str:
        return self.strftime("%Y-%m-%d %H:%M:%S")

    def to_day_datetime_string(self) -> str:
        return self.strftime("%a, %b %d, %Y %I:%M %p")

    def to_time_string(self) -> str:
        return self.strftime("%H:%M:%S")

    def to_time_string_with_seconds(self) -> str:
        return self.strftime("%H:%M:%S")

    def __add__(self, other: Any) -> Any:
        if isinstance(other, datetime.timedelta):
            return self._with_timedelta(other)
        return datetime.datetime.__add__(self, other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, datetime.timedelta):
            return self._with_timedelta(-other)
        return datetime.datetime.__sub__(self, other)


DateTime.EPOCH = DateTime(1970, 1, 1, tzinfo=UTC)
DateTime.min = DateTime(1, 1, 1)
DateTime.max = DateTime(9999, 12, 31, 23, 59, 59, 999999)