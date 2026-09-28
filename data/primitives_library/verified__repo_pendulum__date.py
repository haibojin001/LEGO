from __future__ import annotations

import calendar
import math
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn
from typing import overload

import pendulum

from pendulum.constants import MONTHS_PER_YEAR
from pendulum.constants import YEARS_PER_CENTURY
from pendulum.constants import YEARS_PER_DECADE
from pendulum.day import WeekDay
from pendulum.exceptions import PendulumException
from pendulum.helpers import add_duration
from pendulum.interval import Interval
from pendulum.mixins.default import FormattableMixin

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import SupportsIndex


class Date(FormattableMixin, date):
    _MODIFIERS_VALID_UNITS: ClassVar[list[str]] = [
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
    ]

    def set(
        self, year: int | None = None, month: int | None = None, day: int | None = None
    ) -> Self:
        return self.replace(year=year, month=month, day=day)

    @property
    def day_of_week(self) -> WeekDay:
        return WeekDay(self.weekday())

    @property
    def day_of_year(self) -> int:
        correction = 1 if self.is_leap_year() else 2
        return (
            (275 * self.month) // 9
            - correction * ((self.month + 9) // 12)
            + self.day
            - 30
        )

    @property
    def week_of_year(self) -> int:
        return self.isocalendar()[1]

    @property
    def days_in_month(self) -> int:
        return calendar.monthrange(self.year, self.month)[1]

    @property
    def week_of_month(self) -> int:
        return math.ceil((self.day + self.first_of("month").isoweekday() - 1) / 7)

    @property
    def age(self) -> int:
        return self.diff(abs=False).in_years()

    @property
    def quarter(self) -> int:
        return math.ceil(self.month / 3)

    def to_date_string(self) -> str:
        return self.strftime("%Y-%m-%d")

    def to_formatted_date_string(self) -> str:
        return self.strftime("%b %d, %Y")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.year}, {self.month}, {self.day})"

    def closest(self, dt1: date, dt2: date) -> Self:
        first = self.__class__(dt1.year, dt1.month, dt1.day)
        second = self.__class__(dt2.year, dt2.month, dt2.day)

        if self.diff(first).in_seconds() < self.diff(second).in_seconds():
            return first

        return second

    def farthest(self, dt1: date, dt2: date) -> Self:
        first = self.__class__(dt1.year, dt1.month, dt1.day)
        second = self.__class__(dt2.year, dt2.month, dt2.day)

        if self.diff(first).in_seconds() > self.diff(second).in_seconds():
            return first

        return second

    def is_future(self) -> bool:
        return self > self.today()

    def is_past(self) -> bool:
        return self < self.today()

    def is_leap_year(self) -> bool:
        return calendar.isleap(self.year)

    def is_long_year(self) -> bool:
        return Date(self.year, 12, 28).isocalendar()[1] == 53

    def is_same_day(self, dt: date) -> bool:
        return self == dt

    def is_anniversary(self, dt: date | None = None) -> bool:
        if dt is None:
            dt = self.__class__.today()

        other = self.__class__(dt.year, dt.month, dt.day)
        return (self.month, self.day) == (other.month, other.day)

    is_birthday = is_anniversary

    def add(
        self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0
    ) -> Self:
        result = add_duration(
            date(self.year, self.month, self.day),
            years=years,
            months=months,
            weeks=weeks,
            days=days,
        )
        return self.__class__(result.year, result.month, result.day)

    def subtract(
        self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0
    ) -> Self:
        return self.add(years=-years, months=-months, weeks=-weeks, days=-days)

    def _add_timedelta(self, delta: timedelta) -> Self:
        if isinstance(delta, pendulum.Duration):
            return self.add(
                years=delta.years,
                months=delta.months,
                weeks=delta.weeks,
                days=delta.remaining_days,
            )

        return self.add(days=delta.days)

    def _subtract_timedelta(self, delta: timedelta) -> Self:
        if isinstance(delta, pendulum.Duration):
            return self.subtract(
                years=delta.years,
                months=delta.months,
                weeks=delta.weeks,
                days=delta.remaining_days,
            )

        return self.subtract(days=delta.days)

    def __add__(self, other: timedelta) -> Self:
        if not isinstance(other, timedelta):
            return NotImplemented

        return self._add_timedelta(other)

    @overload
    def __sub__(self, __delta: timedelta) -> Self: ...

    @overload
    def __sub__(self, __dt: datetime) -> NoReturn: ...

    @overload
    def __sub__(self, __dt: Self) -> Interval[Date]: ...

    def __sub__(self, other: timedelta | date) -> Self | Interval[Date]:
        if isinstance(other, timedelta):
            return self._subtract_timedelta(other)

        if not isinstance(other, date):
            return NotImplemented

        other_date = self.__class__(other.year, other.month, other.day)
        return other_date.diff(self, False)

    def diff(self, dt: date | None = None, abs: bool = True) -> Interval[Date]:
        if dt is None:
            dt = self.today()

        return Interval(self, Date(dt.year, dt.month, dt.day), absolute=abs)

    def diff_for_humans(
        self,
        other: date | None = None,
        absolute: bool = False,
        locale: str | None = None,
    ) -> str:
        return self.diff(other, abs=absolute).in_words(locale=locale)

    def _validate_modifier_unit(self, unit: str) -> None:
        if unit not in self._MODIFIERS_VALID_UNITS:
            raise ValueError(f"Invalid unit {unit}.")

    def start_of(self, unit: str) -> Self:
        self._validate_modifier_unit(unit)

        if unit == "day":
            return self

        if unit == "week":
            return self.subtract(days=int(self.day_of_week))

        if unit == "month":
            return self.__class__(self.year, self.month, 1)

        if unit == "year":
            return self.__class__(self.year, 1, 1)

        if unit == "decade":
            year = self.year - self.year % YEARS_PER_DECADE
            return self.__class__(year, 1, 1)

        year = self.year - self.year % YEARS_PER_CENTURY + 1
        return self.__class__(year, 1, 1)

    def end_of(self, unit: str) -> Self:
        self._validate_modifier_unit(unit)

        if unit == "day":
            return self

        if unit == "week":
            return self.add(days=6 - int(self.day_of_week))

        if unit == "month":
            return self.__class__(self.year, self.month, self.days_in_month)

        if unit == "year":
            return self.__class__(self.year, MONTHS_PER_YEAR, 31)

        if unit == "decade":
            year = self.year - self.year % YEARS_PER_DECADE + YEARS_PER_DECADE - 1
            return self.__class__(year, MONTHS_PER_YEAR, 31)

        year = self.year - self.year % YEARS_PER_CENTURY + YEARS_PER_CENTURY
        return self.__class__(year, MONTHS_PER_YEAR, 31)

    def first_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        self._validate_modifier_unit(unit)
        result = self.start_of(unit)

        if day_of_week is None:
            return result

        target = WeekDay(day_of_week)
        offset = (int(target) - int(result.day_of_week)) % 7
        return result.add(days=offset)

    def last_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        self._validate_modifier_unit(unit)
        result = self.end_of(unit)

        if day_of_week is None:
            return result

        target = WeekDay(day_of_week)
        offset = (int(result.day_of_week) - int(target)) % 7
        return result.subtract(days=offset)

    def nth_of(self, unit: str, nth: int, day_of_week: WeekDay) -> Self:
        self._validate_modifier_unit(unit)

        if nth <= 0:
            raise PendulumException(
                "The given day of week does not exist in the scope of the given unit."
            )

        result = self.first_of(unit, day_of_week).add(weeks=nth - 1)
        if result > self.end_of(unit):
            raise PendulumException(
                "The given day of week does not exist in the scope of the given unit."
            )

        return result

    def next(self, day_of_week: WeekDay | None = None) -> Self:
        if day_of_week is None:
            return self.add(days=1)

        target = WeekDay(day_of_week)
        offset = (int(target) - int(self.day_of_week)) % 7
        if offset == 0:
            offset = 7

        return self.add(days=offset)

    def previous(self, day_of_week: WeekDay | None = None) -> Self:
        if day_of_week is None:
            return self.subtract(days=1)

        target = WeekDay(day_of_week)
        offset = (int(self.day_of_week) - int(target)) % 7
        if offset == 0:
            offset = 7

        return self.subtract(days=offset)

    def replace(
        self,
        year: SupportsIndex | None = None,
        month: SupportsIndex | None = None,
        day: SupportsIndex | None = None,
    ) -> Self:
        return self.__class__(
            self.year if year is None else year,
            self.month if month is None else month,
            self.day if day is None else day,
        )


Date.min = Date(1, 1, 1)
Date.max = Date(9999, 12, 31)
Date.EPOCH = Date(1970, 1, 1)