from __future__ import annotations

import datetime
from datetime import time
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import SECS_PER_HOUR
from pendulum.constants import SECS_PER_MIN
from pendulum.constants import USECS_PER_SEC
from pendulum.duration import Duration
from pendulum.mixins.default import FormattableMixin
from pendulum.tz.timezone import UTC

try:
    from pendulum.duration import AbsoluteDuration
except ImportError:
    AbsoluteDuration = Duration
    _HAS_ABSOLUTE_DURATION = False
else:
    _HAS_ABSOLUTE_DURATION = True

if TYPE_CHECKING:
    from typing import Literal

    from typing_extensions import Self
    from typing_extensions import SupportsIndex

    from pendulum.tz.timezone import FixedTimezone
    from pendulum.tz.timezone import Timezone


class Time(FormattableMixin, time):
    """
    Represents a time instance as hour, minute, second, microsecond.
    """

    @classmethod
    def instance(
        cls,
        t: time,
        tz: str | Timezone | FixedTimezone | datetime.tzinfo | None = UTC,
    ) -> Self:
        timezone = t.tzinfo or tz

        if timezone is not None:
            timezone = pendulum._safe_timezone(timezone)

        return cls(
            t.hour,
            t.minute,
            t.second,
            t.microsecond,
            tzinfo=timezone,
            fold=t.fold,
        )

    def __repr__(self) -> str:
        microsecond = f", {self.microsecond}" if self.microsecond else ""
        timezone = f", tzinfo={self.tzinfo!r}" if self.tzinfo else ""

        return (
            f"{self.__class__.__name__}"
            f"({self.hour}, {self.minute}, {self.second}{microsecond}{timezone})"
        )

    def closest(self, dt1: Time | time, dt2: Time | time) -> Self:
        first = self.__class__(
            dt1.hour, dt1.minute, dt1.second, dt1.microsecond
        )
        second = self.__class__(
            dt2.hour, dt2.minute, dt2.second, dt2.microsecond
        )

        if self.diff(first).in_seconds() < self.diff(second).in_seconds():
            return first

        return second

    def farthest(self, dt1: Time | time, dt2: Time | time) -> Self:
        first = self.__class__(
            dt1.hour, dt1.minute, dt1.second, dt1.microsecond
        )
        second = self.__class__(
            dt2.hour, dt2.minute, dt2.second, dt2.microsecond
        )

        if self.diff(first).in_seconds() > self.diff(second).in_seconds():
            return first

        return second

    def add(
        self,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Time:
        from pendulum.datetime import DateTime

        return (
            DateTime.EPOCH.at(self.hour, self.minute, self.second, self.microsecond)
            .add(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
            )
            .time()
        )

    def subtract(
        self,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Time:
        from pendulum.datetime import DateTime

        return (
            DateTime.EPOCH.at(self.hour, self.minute, self.second, self.microsecond)
            .subtract(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
            )
            .time()
        )

    def add_timedelta(self, delta: datetime.timedelta) -> Time:
        if delta.days:
            raise TypeError("Cannot add timedelta with days to Time.")

        return self.add(seconds=delta.seconds, microseconds=delta.microseconds)

    def subtract_timedelta(self, delta: datetime.timedelta) -> Time:
        if delta.days:
            raise TypeError("Cannot subtract timedelta with days to Time.")

        return self.subtract(seconds=delta.seconds, microseconds=delta.microseconds)

    def __add__(self, other: datetime.timedelta) -> Time:
        if not isinstance(other, timedelta):
            return NotImplemented

        return self.add_timedelta(other)

    @overload
    def __sub__(self, other: time) -> pendulum.Duration: ...

    @overload
    def __sub__(self, other: datetime.timedelta) -> Time: ...

    def __sub__(self, other: time | datetime.timedelta) -> pendulum.Duration | Time:
        if not isinstance(other, (Time, time, timedelta)):
            return NotImplemented

        if isinstance(other, timedelta):
            return self.subtract_timedelta(other)

        if isinstance(other, time):
            if other.tzinfo is not None:
                raise TypeError("Cannot subtract aware times to or from Time.")

            other = self.__class__(
                other.hour,
                other.minute,
                other.second,
                other.microsecond,
            )

        return other.diff(self, False)

    @overload
    def __rsub__(self, other: time) -> pendulum.Duration: ...

    @overload
    def __rsub__(self, other: datetime.timedelta) -> Time: ...

    def __rsub__(self, other: time | datetime.timedelta) -> pendulum.Duration | Time:
        if not isinstance(other, (Time, time)):
            return NotImplemented

        if isinstance(other, time):
            if other.tzinfo is not None:
                raise TypeError("Cannot subtract aware times to or from Time.")

            other = self.__class__(
                other.hour,
                other.minute,
                other.second,
                other.microsecond,
            )

        return other.__sub__(self)

    def diff(self, dt: time | None = None, abs: bool = True) -> Duration:
        if dt is None:
            target = pendulum.now().time()
        else:
            target = self.__class__(
                dt.hour,
                dt.minute,
                dt.second,
                dt.microsecond,
            )

        current_microseconds = (
            (
                self.hour * SECS_PER_HOUR
                + self.minute * SECS_PER_MIN
                + self.second
            )
            * USECS_PER_SEC
            + self.microsecond
        )
        target_microseconds = (
            (
                target.hour * SECS_PER_HOUR
                + target.minute * SECS_PER_MIN
                + target.second
            )
            * USECS_PER_SEC
            + target.microsecond
        )

        delta = target_microseconds - current_microseconds
        if abs:
            if _HAS_ABSOLUTE_DURATION:
                return AbsoluteDuration(microseconds=delta)
            return Duration(microseconds=-delta if delta < 0 else delta)

        return Duration(microseconds=delta)

    def diff_for_humans(
        self,
        other: time | None = None,
        absolute: bool = False,
        locale: str | None = None,
    ) -> str:
        now = other is None
        if now:
            other = pendulum.now().time()

        return pendulum.format_diff(self.diff(other), now, absolute, locale)

    def replace(
        self,
        hour: SupportsIndex | None = None,
        minute: SupportsIndex | None = None,
        second: SupportsIndex | None = None,
        microsecond: SupportsIndex | None = None,
        tzinfo: bool | datetime.tzinfo | Literal[True] | None = True,
        fold: int = 0,
    ) -> Self:
        if tzinfo is True:
            tzinfo = self.tzinfo

        actual_hour = self.hour if hour is None else hour
        actual_minute = self.minute if minute is None else minute
        actual_second = self.second if second is None else second
        actual_microsecond = self.microsecond if microsecond is None else microsecond

        replaced = super().replace(
            actual_hour,
            actual_minute,
            actual_second,
            actual_microsecond,
            tzinfo=cast("datetime.tzinfo | None", tzinfo),
            fold=fold,
        )

        return self.__class__(
            replaced.hour,
            replaced.minute,
            replaced.second,
            replaced.microsecond,
            tzinfo=replaced.tzinfo,
            fold=replaced.fold,
        )

    def __getnewargs__(self) -> tuple[Time]:
        return (self,)

    def _get_state(
        self,
        protocol: SupportsIndex = 3,
    ) -> tuple[int, int, int, int, datetime.tzinfo | None]:
        return (
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.tzinfo,
        )

    def __reduce_ex__(self, protocol: SupportsIndex):
        return self.__class__, self._get_state(protocol)