from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from typing import overload

import pendulum

from pendulum.constants import SECONDS_PER_DAY
from pendulum.constants import SECONDS_PER_HOUR
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.constants import US_PER_SECOND
from pendulum.utils._compat import PYPY

if TYPE_CHECKING:
    from typing_extensions import Self


def _divide_and_round(a: float, b: float) -> int:
    """Divide a by b and round to the nearest integer using ties-to-even."""
    quotient, remainder = divmod(a, b)
    quotient = int(quotient)
    remainder *= 2

    exceeds_half = remainder > b if b > 0 else remainder < b
    if exceeds_half or (remainder == b and quotient % 2):
        quotient += 1

    return quotient


class Duration(timedelta):
    """Replacement for the standard :class:`datetime.timedelta` class."""

    _total: float = 0
    _years: int = 0
    _months: int = 0
    _weeks: int = 0
    _days: int = 0
    _remaining_days: int = 0
    _seconds: int = 0
    _microseconds: int = 0

    _y = None
    _m = None
    _w = None
    _d = None
    _h = None
    _i = None
    _s = None
    _invert = None

    def __new__(
        cls,
        days: float = 0,
        seconds: float = 0,
        microseconds: float = 0,
        milliseconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        weeks: float = 0,
        years: float = 0,
        months: float = 0,
    ) -> Self:
        if not isinstance(years, int) or not isinstance(months, int):
            raise ValueError("Float year and months are not supported")

        calendar_days = years * 365 + months * 30
        instance = timedelta.__new__(
            cls,
            days + calendar_days,
            seconds,
            microseconds,
            milliseconds,
            minutes,
            hours,
            weeks,
        )

        value = instance.total_seconds() - calendar_days * SECONDS_PER_DAY
        instance._total = value

        direction = -1 if value < 0 else 1
        instance._microseconds = round(value % direction * US_PER_SECOND)
        instance._seconds = abs(int(value)) % SECONDS_PER_DAY * direction

        ordinary_days = abs(int(value)) // SECONDS_PER_DAY * direction
        instance._days = ordinary_days
        instance._remaining_days = abs(ordinary_days) % 7 * direction
        instance._weeks = abs(ordinary_days) // 7 * direction
        instance._months = months
        instance._years = years

        instance._signature = {
            "years": years,
            "months": months,
            "weeks": weeks,
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "microseconds": microseconds + milliseconds * 1000,
        }

        return instance

    def total_minutes(self) -> float:
        return self.total_seconds() / SECONDS_PER_MINUTE

    def total_hours(self) -> float:
        return self.total_seconds() / SECONDS_PER_HOUR

    def total_days(self) -> float:
        return self.total_seconds() / SECONDS_PER_DAY

    def total_weeks(self) -> float:
        return self.total_days() / 7

    if PYPY:

        def total_seconds(self) -> float:
            days = 0

            if hasattr(self, "_years"):
                days += self._years * 365

            if hasattr(self, "_months"):
                days += self._months * 30

            if hasattr(self, "_remaining_days"):
                days += self._weeks * 7 + self._remaining_days
            else:
                days += self._days

            return (
                (days * SECONDS_PER_DAY + self._seconds) * US_PER_SECOND
                + self._microseconds
            ) / US_PER_SECOND

    @property
    def years(self) -> int:
        return self._years

    @property
    def months(self) -> int:
        return self._months

    @property
    def weeks(self) -> int:
        return self._weeks

    if PYPY:

        @property
        def days(self) -> int:
            return self._years * 365 + self._months * 30 + self._days

    @property
    def remaining_days(self) -> int:
        return self._remaining_days

    @property
    def hours(self) -> int:
        if self._h is None:
            self._h = 0
            if abs(self._seconds) >= SECONDS_PER_HOUR:
                self._h = (
                    abs(self._seconds) // SECONDS_PER_HOUR % 24
                ) * self._sign(self._seconds)

        return self._h

    @property
    def minutes(self) -> int:
        if self._i is None:
            self._i = 0
            if abs(self._seconds) >= SECONDS_PER_MINUTE:
                self._i = (
                    abs(self._seconds) // SECONDS_PER_MINUTE % 60
                ) * self._sign(self._seconds)

        return self._i

    @property
    def seconds(self) -> int:
        return self._seconds

    @property
    def remaining_seconds(self) -> int:
        if self._s is None:
            self._s = abs(self._seconds) % 60 * self._sign(self._seconds)

        return self._s

    @property
    def microseconds(self) -> int:
        return self._microseconds

    @property
    def invert(self) -> bool:
        if self._invert is None:
            self._invert = self.total_seconds() < 0

        return self._invert

    def in_weeks(self) -> int:
        return int(self.total_weeks())

    def in_days(self) -> int:
        return int(self.total_days())

    def in_hours(self) -> int:
        return int(self.total_hours())

    def in_minutes(self) -> int:
        return int(self.total_minutes())

    def in_seconds(self) -> int:
        return int(self.total_seconds())

    def in_words(self, locale: str | None = None, separator: str = " ") -> str:
        values = (
            ("year", self.years),
            ("month", self.months),
            ("week", self.weeks),
            ("day", self.remaining_days),
            ("hour", self.hours),
            ("minute", self.minutes),
            ("second", self.remaining_seconds),
        )

        if locale is None:
            locale = pendulum.get_locale()

        locale_data = pendulum.locale(locale)
        words: list[str] = []

        for name, amount in values:
            if abs(amount):
                key = f"units.{name}.{locale_data.plural(abs(amount))}"
                words.append(locale_data.translation(key).format(amount))

        if not words:
            amount: int | str = 0
            if self.microseconds:
                key = f"units.second.{locale_data.plural(0)}"
                amount = f"{abs(self.microseconds) / US_PER_SECOND:.2f}"
            else:
                key = f"units.microsecond.{locale_data.plural(0)}"

            words.append(locale_data.translation(key).format(amount))

        return separator.join(words)

    def _sign(self, value: float) -> int:
        return -1 if value < 0 else 1

    def as_timedelta(self) -> timedelta:
        return timedelta(seconds=self.total_seconds())

    def __str__(self) -> str:
        return self.in_words()

    def __repr__(self) -> str:
        result = f"{self.__class__.__name__}("

        if self._years:
            result += f"years={self._years}, "
        if self._months:
            result += f"months={self._months}, "
        if self._weeks:
            result += f"weeks={self._weeks}, "
        if self._days:
            result += f"days={self._remaining_days}, "
        if self.hours:
            result += f"hours={self.hours}, "
        if self.minutes:
            result += f"minutes={self.minutes}, "
        if self.remaining_seconds:
            result += f"seconds={self.remaining_seconds}, "
        if self.microseconds:
            result += f"microseconds={self.microseconds}, "

        return (result + ")").replace(", )", ")")

    def __add__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            return self.__class__(
                seconds=self.total_seconds() + other.total_seconds()
            )

        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            return self.__class__(
                seconds=self.total_seconds() - other.total_seconds()
            )

        return NotImplemented

    def __rsub__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            return self.__class__(
                seconds=other.total_seconds() - self.total_seconds()
            )

        return NotImplemented

    def __mul__(self, other: int | float) -> Self:
        if isinstance(other, int):
            return self.__class__(seconds=self.total_seconds() * other)

        if isinstance(other, float):
            return self.__class__(
                microseconds=_divide_and_round(
                    self.total_seconds() * US_PER_SECOND, 1.0 / other
                )
            )

        return NotImplemented

    __rmul__ = __mul__

    @overload
    def __truediv__(self, other: timedelta) -> float:
        ...

    @overload
    def __truediv__(self, other: int | float) -> Self:
        ...

    def __truediv__(self, other: timedelta | int | float) -> Self | float:
        if isinstance(other, timedelta):
            return self.total_seconds() / other.total_seconds()

        if isinstance(other, (int, float)):
            return self.__class__(
                microseconds=_divide_and_round(
                    self.total_seconds() * US_PER_SECOND, other
                )
            )

        return NotImplemented

    @overload
    def __floordiv__(self, other: timedelta) -> int:
        ...

    @overload
    def __floordiv__(self, other: int | float) -> Self:
        ...

    def __floordiv__(self, other: timedelta | int | float) -> Self | int:
        if isinstance(other, timedelta):
            return self.total_seconds() // other.total_seconds()

        if isinstance(other, (int, float)):
            return self.__class__(
                microseconds=(self.total_seconds() * US_PER_SECOND) // other
            )

        return NotImplemented

    def __mod__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            return self.__class__(
                microseconds=(
                    self.total_seconds() * US_PER_SECOND
                ) % (other.total_seconds() * US_PER_SECOND)
            )

        return NotImplemented

    def __divmod__(self, other: timedelta) -> tuple[int, Self]:
        if isinstance(other, timedelta):
            quotient, remainder = divmod(
                self.total_seconds() * US_PER_SECOND,
                other.total_seconds() * US_PER_SECOND,
            )
            return int(quotient), self.__class__(microseconds=remainder)

        return NotImplemented

    def __pos__(self) -> Self:
        return self.__class__(seconds=self.total_seconds())

    def __neg__(self) -> Self:
        return self.__class__(seconds=-self.total_seconds())

    def __abs__(self) -> Self:
        return self.__class__(seconds=abs(self.total_seconds()))


class AbsoluteDuration(Duration):
    """A duration representing an absolute elapsed amount of time."""

    pass