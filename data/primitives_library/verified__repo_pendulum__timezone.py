from __future__ import annotations

import datetime as _datetime
import zoneinfo

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, cast

from pendulum.tz.exceptions import AmbiguousTime, InvalidTimezone, NonExistingTime

if TYPE_CHECKING:
    from typing_extensions import Self


POST_TRANSITION = "post"
PRE_TRANSITION = "pre"
TRANSITION_ERROR = "error"

_DT = TypeVar("_DT", bound=_datetime.datetime)


class PendulumTimezone(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def convert(self, dt: _DT, raise_on_unknown_times: bool = False) -> _DT:
        raise NotImplementedError

    @abstractmethod
    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> _datetime.datetime:
        raise NotImplementedError


class Timezone(zoneinfo.ZoneInfo, PendulumTimezone):
    def __new__(cls, key: str) -> Self:
        try:
            return super().__new__(cls, key)
        except zoneinfo.ZoneInfoNotFoundError:
            raise InvalidTimezone(key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timezone) and self.key == other.key

    @property
    def name(self) -> str:
        return self.key

    def convert(self, dt: _DT, raise_on_unknown_times: bool = False) -> _DT:
        if dt.tzinfo is not None:
            return cast(_DT, dt.astimezone(self))

        if dt.fold:
            before = cast(_datetime.timedelta, self.utcoffset(dt.replace(fold=0)))
            after = cast(_datetime.timedelta, self.utcoffset(dt))
        else:
            before = cast(_datetime.timedelta, self.utcoffset(dt))
            after = cast(_datetime.timedelta, self.utcoffset(dt.replace(fold=1)))

        if after > before:
            if raise_on_unknown_times:
                raise NonExistingTime(dt)

            adjustment = after - before
            if not dt.fold:
                adjustment = -adjustment

            dt = cast(_DT, dt + adjustment)
        elif before > after and raise_on_unknown_times:
            raise AmbiguousTime(dt)

        return dt.replace(tzinfo=self)

    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> _datetime.datetime:
        value = _datetime.datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            fold=1,
        )
        return self.convert(value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.name}')"


class FixedTimezone(_datetime.tzinfo, PendulumTimezone):
    def __init__(self, offset: int, name: str | None = None) -> None:
        sign = "-" if offset < 0 else "+"
        total_minutes = abs(int(offset / 60))
        hours, minutes = divmod(total_minutes, 60)

        if not name:
            name = f"{sign}{hours:02d}:{minutes:02d}"

        self._offset = offset
        self._name = name
        self._utcoffset = _datetime.timedelta(seconds=offset)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FixedTimezone) and self._offset == other._offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def offset(self) -> int:
        return self._offset

    def convert(self, dt: _DT, raise_on_unknown_times: bool = False) -> _DT:
        if dt.tzinfo is not None:
            return cast(_DT, dt.astimezone(self))

        return dt.__class__(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=self,
            fold=0,
        )

    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> _datetime.datetime:
        value = _datetime.datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            fold=1,
        )
        return self.convert(value)

    def utcoffset(self, dt: _datetime.datetime | None) -> _datetime.timedelta:
        return self._utcoffset

    def dst(self, dt: _datetime.datetime | None) -> _datetime.timedelta:
        return _datetime.timedelta()

    def fromutc(self, dt: _datetime.datetime) -> _datetime.datetime:
        value = _datetime.datetime.__add__(dt, self._utcoffset)
        return value.replace(tzinfo=self)

    def tzname(self, dt: _datetime.datetime | None) -> str | None:
        return self._name

    def __getinitargs__(self) -> tuple[int, str]:
        return self._offset, self._name

    def __repr__(self) -> str:
        suffix = f', name="{self._name}"' if self._name else ""
        return f"{type(self).__name__}({self._offset}{suffix})"


UTC = Timezone("UTC")