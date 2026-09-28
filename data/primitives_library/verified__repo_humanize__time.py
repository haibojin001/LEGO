from __future__ import annotations

__lazy_modules__ = {"humanize.i18n", "humanize.number"}

from enum import Enum
from functools import total_ordering

from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma

TYPE_CHECKING = False
if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable
    from typing import Any

__all__ = [
    "naturaldate",
    "naturalday",
    "naturaldelta",
    "naturaltime",
    "precisedelta",
]


@total_ordering
class Unit(Enum):
    MICROSECONDS = 0
    MILLISECONDS = 1
    SECONDS = 2
    MINUTES = 3
    HOURS = 4
    DAYS = 5
    MONTHS = 6
    YEARS = 7

    def __lt__(self, other):
        if type(self) is type(other):
            return self.value < other.value
        return NotImplemented


def _now():
    import datetime as datetime

    return datetime.datetime.now()


def _abs_timedelta(delta):
    if delta.days < 0:
        current = _now()
        return current - (current + delta)
    return delta


def _date_and_delta(value, *, now=None, precise=False):
    import datetime as datetime

    if not now:
        now = _now()

    if isinstance(value, datetime.datetime):
        date = value
        delta = now - value
    elif isinstance(value, datetime.timedelta):
        date = now - value
        delta = value
    else:
        try:
            seconds = value if precise else round(value)
            delta = datetime.timedelta(seconds=seconds)
            date = now - delta
        except (ValueError, TypeError):
            return None, value

    return date, _abs_timedelta(delta)


def _convert_aware_datetime(value):
    import datetime as datetime

    if not isinstance(value, datetime.datetime) or value.tzinfo is None:
        return value

    offset = value.utcoffset()
    if offset is None:
        return value

    return value.replace(tzinfo=None) - offset


def naturaldelta(value, months=True, minimum_unit="seconds"):
    import datetime as datetime
    import math

    unit = Unit[minimum_unit.upper()]
    if unit not in (Unit.SECONDS, Unit.MILLISECONDS, Unit.MICROSECONDS):
        raise ValueError(f"Minimum unit '{minimum_unit}' not supported")

    if isinstance(value, datetime.timedelta):
        delta = value
    else:
        try:
            int(value)
            value = float(value)
            delta = datetime.timedelta(seconds=value)
        except (ValueError, TypeError):
            return str(value)
        except OverflowError:
            try:
                finite = math.isfinite(value)
            except (TypeError, ValueError):
                return str(value)
            if not finite:
                return str(value)
            raise

    delta = abs(delta)
    years = delta.days // 365
    days = delta.days % 365
    number_of_months = round(days / 30.5)

    if years == 0 and days < 1:
        if delta.seconds == 0:
            if unit == Unit.MICROSECONDS and delta.microseconds < 1000:
                return (
                    _ngettext("%d microsecond", "%d microseconds", delta.microseconds)
                    % delta.microseconds
                )

            if unit == Unit.MILLISECONDS or (
                unit == Unit.MICROSECONDS
                and 1000 <= delta.microseconds < 1_000_000
            ):
                milliseconds = delta.microseconds / 1000
                return (
                    _ngettext("%d millisecond", "%d milliseconds", int(milliseconds))
                    % milliseconds
                )

            return _("a moment")

        if delta.seconds == 1:
            return _("a second")

        if delta.seconds < 60:
            return _ngettext("%d second", "%d seconds", delta.seconds) % delta.seconds

        if delta.seconds < 3600:
            minutes = round(delta.seconds / 60)
            if minutes == 1:
                return _("a minute")
            if minutes == 60:
                return _("an hour")
            return _ngettext("%d minute", "%d minutes", minutes) % minutes

        hours = round(delta.seconds / 3600)
        if hours == 1:
            return _("an hour")
        if hours == 24:
            return _("a day")
        return _ngettext("%d hour", "%d hours", hours) % hours

    if years == 0:
        if days == 1:
            return _("a day")

        if not months or number_of_months == 0:
            return _ngettext("%d day", "%d days", days) % days

        if number_of_months == 1:
            return _("a month")

        if number_of_months == 12:
            return _("a year")

        return _ngettext("%d month", "%d months", number_of_months) % number_of_months

    if years == 1:
        if number_of_months == 0 and days == 0:
            return _("a year")

        if number_of_months == 0:
            return _ngettext("1 year, %d day", "1 year, %d days", days) % days

        if months:
            if number_of_months == 1:
                return _("1 year, 1 month")

            if number_of_months == 12:
                years += 1
                return _ngettext("%d year", "%d years", years) % years

            return (
                _ngettext("1 year, %d month", "1 year, %d months", number_of_months)
                % number_of_months
            )

        return _ngettext("1 year, %d day", "1 year, %d days", days) % days

    return _ngettext("%d year", "%d years", years).replace("%d", "%s") % intcomma(
        years
    )


def naturaltime(
    value,
    future=False,
    months=True,
    minimum_unit="seconds",
    when=None,
):
    import datetime as datetime

    value = _convert_aware_datetime(value)
    when = _convert_aware_datetime(when)
    current = when or _now()

    date, delta = _date_and_delta(value, now=current)
    if date is None:
        return str(value)

    if isinstance(value, (datetime.datetime, datetime.timedelta)):
        future = date > current

    text = naturaldelta(delta, months=months, minimum_unit=minimum_unit)
    if future:
        return _("%s from now") % text
    return _("%s ago") % text


def naturalday(value, format="%b %d"):
    import datetime as datetime

    try:
        value = datetime.date(value.year, value.month, value.day)
    except (AttributeError, TypeError):
        return str(value)

    difference = value - datetime.date.today()

    if difference.days == 0:
        return _("today")
    if difference.days == 1:
        return _("tomorrow")
    if difference.days == -1:
        return _("yesterday")

    return value.strftime(format)


def naturaldate(value, format="%b %d"):
    import datetime as datetime

    try:
        if value.year == datetime.date.today().year:
            return naturalday(value, format)
        return value.strftime("%b %d %Y")
    except AttributeError:
        return str(value)


def precisedelta(value, minimum_unit="seconds", suppress=(), format="%0.2f"):
    import datetime as datetime

    minimum = Unit[minimum_unit.upper()]
    suppressed = [Unit[item.upper()] for item in suppress]

    if minimum in suppressed:
        raise ValueError("Minimum unit cannot be suppressed")

    date, delta = _date_and_delta(value, precise=True)
    if date is None:
        return str(value)

    delta = abs(delta)

    scales = {
        Unit.YEARS: 365 * 24 * 60 * 60 * 1_000_000,
        Unit.MONTHS: 30 * 24 * 60 * 60 * 1_000_000,
        Unit.DAYS: 24 * 60 * 60 * 1_000_000,
        Unit.HOURS: 60 * 60 * 1_000_000,
        Unit.MINUTES: 60 * 1_000_000,
        Unit.SECONDS: 1_000_000,
        Unit.MILLISECONDS: 1_000,
        Unit.MICROSECONDS: 1,
    }

    names = {
        Unit.YEARS: ("%d year", "%d years"),
        Unit.MONTHS: ("%d month", "%d months"),
        Unit.DAYS: ("%d day", "%d days"),
        Unit.HOURS: ("%d hour", "%d hours"),
        Unit.MINUTES: ("%d minute", "%d minutes"),
        Unit.SECONDS: ("%s second", "%s seconds"),
        Unit.MILLISECONDS: ("%d millisecond", "%d milliseconds"),
        Unit.MICROSECONDS: ("%d microsecond", "%d microseconds"),
    }

    remaining = (
        delta.days * 24 * 60 * 60 * 1_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    pieces = []

    for unit in reversed(Unit):
        if unit < minimum:
            continue
        if unit in suppressed:
            continue

        amount, remaining = divmod(remaining, scales[unit])

        if unit == minimum:
            if unit == Unit.SECONDS:
                rendered = format % (amount + remaining / 1_000_000)
                count = amount + remaining / 1_000_000
            else:
                rendered = str(amount)
                count = amount

            singular, plural = names[unit]
            pieces.append(_ngettext(singular, plural, count) % rendered)
            break

        if amount:
            singular, plural = names[unit]
            pieces.append(_ngettext(singular, plural, amount) % amount)

    if not pieces:
        singular, plural = names[minimum]
        if minimum == Unit.SECONDS:
            rendered = format % 0
            return _ngettext(singular, plural, 0) % rendered
        return _ngettext(singular, plural, 0) % 0

    if len(pieces) == 1:
        return pieces[0]

    return _(", ").join(pieces[:-1]) + " " + _("and") + " " + pieces[-1]