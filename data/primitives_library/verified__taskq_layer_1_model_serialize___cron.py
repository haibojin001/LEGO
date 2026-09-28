import calendar
import datetime
import math
import time
from typing import FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple


class CronExpr:
    def __init__(self, expr: str):
        # 5 fields: minute hour day month weekday. Each supports '*', 'a',
        # 'a-b', 'a-b/s', '*/s', comma-separated combinations. Weekday 0..6
        # with 0=Monday (ISO). Raises ValueError on invalid expressions.
        if not isinstance(expr, str):
            raise ValueError("cron expression must be a string")

        parts = expr.split()
        if len(parts) != 5:
            raise ValueError("cron expression must have exactly 5 fields")

        minute = _parse_field(parts[0], 0, 59, "minute")
        hour = _parse_field(parts[1], 0, 23, "hour")
        day = _parse_field(parts[2], 1, 31, "day")
        month = _parse_field(parts[3], 1, 12, "month")
        weekday = _parse_field(parts[4], 0, 6, "weekday")

        self.minutes: FrozenSet[int] = frozenset(minute)
        self.hours: FrozenSet[int] = frozenset(hour)
        self.days: FrozenSet[int] = frozenset(day)
        self.months: FrozenSet[int] = frozenset(month)
        self.weekdays: FrozenSet[int] = frozenset(weekday)

        self._minutes: Tuple[int, ...] = tuple(sorted(self.minutes))
        self._hours: Tuple[int, ...] = tuple(sorted(self.hours))
        self._days: Tuple[int, ...] = tuple(sorted(self.days))
        self._months: Tuple[int, ...] = tuple(sorted(self.months))
        self._weekdays: Tuple[int, ...] = tuple(sorted(self.weekdays))

        self._min_minute = self._minutes[0]
        self._min_hour = self._hours[0]
        self._min_month = self._months[0]

    def next_after(self, after: float) -> float:
        # smallest unix time strictly > after (seconds precision, seconds
        # field == 0) at which all five fields match. Uses local time
        # interpretation via datetime.datetime.fromtimestamp / mktime.
        if not isinstance(after, (int, float)) or not math.isfinite(after):
            raise ValueError("after must be a finite unix timestamp")

        scanned = self._scan_actual_minutes(after, 3 * 60 * 60)
        if scanned is not None:
            return scanned

        try:
            local = datetime.datetime.fromtimestamp(after)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError("after is outside the supported timestamp range") from exc

        current = local.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        limit_year = min(current.year + 400, 9999)

        while current.year <= limit_year:
            candidate = self._next_local_candidate(current, limit_year)

            valid_times: List[float] = []
            for ts in _possible_timestamps_for_local(candidate):
                if ts <= after:
                    continue
                try:
                    back = datetime.datetime.fromtimestamp(ts)
                except (OverflowError, OSError, ValueError):
                    continue
                if (
                    back.year == candidate.year
                    and back.month == candidate.month
                    and back.day == candidate.day
                    and back.hour == candidate.hour
                    and back.minute == candidate.minute
                    and back.second == 0
                    and self._matches_datetime(back)
                ):
                    valid_times.append(float(ts))

            if valid_times:
                return min(valid_times)

            current = candidate + datetime.timedelta(minutes=1)

        raise ValueError("cron expression has no future fire time in supported range")

    def _scan_actual_minutes(self, after: float, span_seconds: int) -> Optional[float]:
        start = math.floor(after / 60.0) * 60.0
        if start <= after:
            start += 60.0
        end = after + span_seconds
        ts = start

        while ts <= end:
            try:
                dt = datetime.datetime.fromtimestamp(ts)
            except (OverflowError, OSError, ValueError):
                return None
            if dt.second == 0 and self._matches_datetime(dt):
                return float(ts)
            ts += 60.0

        return None

    def _matches_datetime(self, dt: datetime.datetime) -> bool:
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.day in self.days
            and dt.month in self.months
            and dt.weekday() in self.weekdays
        )

    def _next_local_candidate(
        self, current: datetime.datetime, limit_year: int
    ) -> datetime.datetime:
        dt = current.replace(second=0, microsecond=0)

        while dt.year <= limit_year:
            month = _next_ge(self._months, dt.month)
            if month is None:
                if dt.year >= 9999:
                    break
                dt = datetime.datetime(
                    dt.year + 1, self._min_month, 1, self._min_hour, self._min_minute
                )
                continue

            if month != dt.month:
                dt = datetime.datetime(
                    dt.year, month, 1, self._min_hour, self._min_minute
                )
                continue

            last_day = calendar.monthrange(dt.year, dt.month)[1]
            if dt.day > last_day:
                dt = self._first_of_next_month(dt)
                continue

            day = self._next_matching_day(dt.year, dt.month, dt.day)
            if day is None:
                dt = self._first_of_next_month(dt)
                continue

            if day != dt.day:
                dt = datetime.datetime(
                    dt.year, dt.month, day, self._min_hour, self._min_minute
                )
                continue

            hour = _next_ge(self._hours, dt.hour)
            if hour is None:
                dt = datetime.datetime(
                    dt.year, dt.month, dt.day, self._min_hour, self._min_minute
                ) + datetime.timedelta(days=1)
                continue

            if hour != dt.hour:
                dt = dt.replace(hour=hour, minute=self._min_minute)
                continue

            minute = _next_ge(self._minutes, dt.minute)
            if minute is None:
                dt = dt.replace(minute=0) + datetime.timedelta(hours=1)
                continue

            if minute != dt.minute:
                dt = dt.replace(minute=minute)
                continue

            return dt

        raise ValueError("cron expression has no future fire time in supported range")

    def _next_matching_day(
        self, year: int, month: int, start_day: int
    ) -> Optional[int]:
        last_day = calendar.monthrange(year, month)[1]
        for day in range(start_day, last_day + 1):
            if day in self.days:
                if datetime.date(year, month, day).weekday() in self.weekdays:
                    return day
        return None

    def _first_of_next_month(self, dt: datetime.datetime) -> datetime.datetime:
        if dt.month == 12:
            if dt.year >= 9999:
                raise ValueError("cron expression has no future fire time in supported range")
            return datetime.datetime(
                dt.year + 1, 1, 1, self._min_hour, self._min_minute
            )
        return datetime.datetime(
            dt.year, dt.month + 1, 1, self._min_hour, self._min_minute
        )


def _parse_field(text: str, minimum: int, maximum: int, name: str) -> Set[int]:
    if text == "":
        raise ValueError(f"empty {name} field")

    values: Set[int] = set()

    for part in text.split(","):
        if part == "":
            raise ValueError(f"empty item in {name} field")

        if part == "*":
            values.update(range(minimum, maximum + 1))
            continue

        if part.startswith("*/"):
            step_text = part[2:]
            step = _parse_positive_int(step_text, f"{name} step")
            values.update(range(minimum, maximum + 1, step))
            continue

        if "/" in part:
            base, step_text = part.split("/", 1)
            if "/" in step_text or "-" not in base:
                raise ValueError(f"invalid stepped range in {name} field")
            step = _parse_positive_int(step_text, f"{name} step")
            start, end = _parse_range(base, minimum, maximum, name)
            values.update(range(start, end + 1, step))
            continue

        if "-" in part:
            start, end = _parse_range(part, minimum, maximum, name)
            values.update(range(start, end + 1))
            continue

        value = _parse_int(part, name)
        _check_bounds(value, minimum, maximum, name)
        values.add(value)

    if not values:
        raise ValueError(f"{name} field matches no values")

    return values


def _parse_range(text: str, minimum: int, maximum: int, name: str) -> Tuple[int, int]:
    pieces = text.split("-")
    if len(pieces) != 2 or pieces[0] == "" or pieces[1] == "":
        raise ValueError(f"invalid range in {name} field")

    start = _parse_int(pieces[0], name)
    end = _parse_int(pieces[1], name)

    _check_bounds(start, minimum, maximum, name)
    _check_bounds(end, minimum, maximum, name)

    if start > end:
        raise ValueError(f"invalid descending range in {name} field")

    return start, end


def _parse_int(text: str, name: str) -> int:
    if not text.isdigit():
        raise ValueError(f"invalid integer in {name} field")
    try:
        return int(text, 10)
    except ValueError as exc:
        raise ValueError(f"invalid integer in {name} field") from exc


def _parse_positive_int(text: str, name: str) -> int:
    value = _parse_int(text, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _check_bounds(value: int, minimum: int, maximum: int, name: str) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} value {value} outside range {minimum}..{maximum}")


def _next_ge(values: Sequence[int], current: int) -> Optional[int]:
    for value in values:
        if value >= current:
            return value
    return None


def _possible_timestamps_for_local(dt: datetime.datetime) -> Tuple[float, ...]:
    seen = set()
    out: List[float] = []

    def add(value: float) -> None:
        key = round(value, 6)
        if key not in seen:
            seen.add(key)
            out.append(float(value))

    try:
        add(time.mktime(dt.timetuple()))
    except (OverflowError, OSError, ValueError):
        pass

    try:
        add(dt.replace(fold=0).timestamp())
    except (OverflowError, OSError, ValueError):
        pass

    try:
        add(dt.replace(fold=1).timestamp())
    except (OverflowError, OSError, ValueError):
        pass

    return tuple(out)