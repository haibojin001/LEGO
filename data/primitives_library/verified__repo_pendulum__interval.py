from __future__ import annotations

import copy
import operator

from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import MONTHS_PER_YEAR
from pendulum.duration import Duration
from pendulum.helpers import precise_diff

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self
    from typing_extensions import SupportsIndex

    from pendulum.helpers import PreciseDiff
    from pendulum.locales.locale import Locale


_T = TypeVar("_T", bound=date)


class Interval(Duration, Generic[_T]):
    """
    A span delimited by two dates or datetimes.
    """

    def __new__(cls, start: _T, end: _T, absolute: bool = False) -> Self:
        start_is_datetime = isinstance(start, datetime)
        end_is_datetime = isinstance(end, datetime)

        if start_is_datetime != end_is_datetime:
            raise ValueError(
                "Both start and end of an Interval must have the same type"
            )

        if start_is_datetime and end_is_datetime:
            if (start.tzinfo is None) != (end.tzinfo is None):
                raise TypeError("can't compare offset-naive and offset-aware datetimes")

        if absolute and start > end:
            start, end = end, start

        def native(value: _T) -> _T:
            if isinstance(value, pendulum.DateTime):
                return cast(
                    _T,
                    datetime(
                        value.year,
                        value.month,
                        value.day,
                        value.hour,
                        value.minute,
                        value.second,
                        value.microsecond,
                        tzinfo=value.tzinfo,
                        fold=value.fold,
                    ),
                )

            if isinstance(value, pendulum.Date):
                return cast(_T, date(value.year, value.month, value.day))

            return value

        left = native(start)
        right = native(end)

        if (
            isinstance(left, datetime)
            and isinstance(right, datetime)
            and left.tzinfo is right.tzinfo
            and left.tzinfo is not None
        ):
            left_offset = cast(timedelta, cast(datetime, start).utcoffset())
            right_offset = cast(timedelta, cast(datetime, end).utcoffset())
            left = cast(_T, (left - left_offset).replace(tzinfo=None))
            right = cast(_T, (right - right_offset).replace(tzinfo=None))

        difference = right - left
        return super().__new__(cls, seconds=difference.total_seconds())

    def __init__(self, start: _T, end: _T, absolute: bool = False) -> None:
        super().__init__()

        def normalize(value: _T) -> tuple[_T, _T]:
            if not isinstance(value, pendulum.Date):
                if isinstance(value, datetime):
                    converted = cast(_T, pendulum.instance(value))
                else:
                    converted = cast(_T, pendulum.date(value.year, value.month, value.day))

                return converted, converted

            if isinstance(value, pendulum.DateTime):
                comparison_value = cast(
                    _T,
                    datetime(
                        value.year,
                        value.month,
                        value.day,
                        value.hour,
                        value.minute,
                        value.second,
                        value.microsecond,
                        tzinfo=value.tzinfo,
                    ),
                )
            else:
                comparison_value = cast(_T, date(value.year, value.month, value.day))

            return value, comparison_value

        actual_start, diff_start = normalize(start)
        actual_end, diff_end = normalize(end)

        self._invert = actual_start > actual_end
        if self._invert and absolute:
            actual_start, actual_end = actual_end, actual_start
            diff_start, diff_end = diff_end, diff_start

        self._absolute = absolute
        self._start = actual_start
        self._end = actual_end
        self._delta: PreciseDiff = precise_diff(diff_start, diff_end)

    @property
    def years(self) -> int:
        return self._delta.years

    @property
    def months(self) -> int:
        return self._delta.months

    @property
    def weeks(self) -> int:
        return abs(self._delta.days) // 7 * self._sign(self._delta.days)

    @property
    def days(self) -> int:
        return self._days

    @property
    def remaining_days(self) -> int:
        return abs(self._delta.days) % 7 * self._sign(self._days)

    @property
    def hours(self) -> int:
        return self._delta.hours

    @property
    def minutes(self) -> int:
        return self._delta.minutes

    @property
    def start(self) -> _T:
        return self._start

    @property
    def end(self) -> _T:
        return self._end

    def in_years(self) -> int:
        """
        Return the count of complete years in this interval.
        """
        return self.years

    def in_months(self) -> int:
        """
        Return the count of complete months in this interval.
        """
        return self.years * MONTHS_PER_YEAR + self.months

    def in_weeks(self) -> int:
        days = self.in_days()
        return (abs(days) // 7) * (-1 if days < 0 else 1)

    def in_days(self) -> int:
        return self._delta.total_days

    def in_words(self, locale: str | None = None, separator: str = " ") -> str:
        """
        Render the interval using localized unit names.
        """
        from pendulum.locales.locale import Locale

        locale_object: Locale = Locale.load(locale or pendulum.get_locale())
        values = (
            ("year", self.years),
            ("month", self.months),
            ("week", self.weeks),
            ("day", self.remaining_days),
            ("hour", self.hours),
            ("minute", self.minutes),
            ("second", self.remaining_seconds),
        )

        parts: list[str] = []
        for unit, value in values:
            if abs(value):
                key = f"units.{unit}.{locale_object.plural(abs(value))}"
                parts.append(locale_object.translation(key).format(value))

        if not parts:
            if abs(self.microseconds):
                value: str | int = f"{abs(self.microseconds) / 1000000:.2f}"
                key = f"units.second.{locale_object.plural(1)}"
            else:
                value = 0
                key = f"units.microsecond.{locale_object.plural(0)}"

            parts.append(locale_object.translation(key).format(value))

        return separator.join(parts)

    def range(self, unit: str, amount: int = 1) -> Iterator[_T]:
        operation = operator.le
        method = "add"

        if not self._absolute and self.invert:
            operation = operator.ge
            method = "subtract"

        current = copy.copy(self.start)
        boundary = copy.copy(self.end)
        increment = amount

        while operation(current, boundary):
            yield current
            current = getattr(self.start, method)(**{unit: increment})
            increment += amount

    def __iter__(self) -> Iterator[_T]:
        return self.range("days")

    @overload
    def __getitem__(self, item: SupportsIndex) -> _T: ...

    @overload
    def __getitem__(self, item: slice) -> list[_T]: ...

    def __getitem__(self, item: SupportsIndex | slice) -> _T | list[_T]:
        values = list(self)
        if isinstance(item, slice):
            return values[item]

        return values[operator.index(item)]

    def __contains__(self, other: Any) -> bool:
        if self.invert:
            return self.end <= other <= self.start

        return self.start <= other <= self.end

    def __repr__(self) -> str:
        return f"<Interval [{self.start} -> {self.end}]>"