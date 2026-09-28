import re
import time
import bisect
import operator
from datetime import tzinfo, timedelta, date, datetime, timezone


total_seconds = timedelta.total_seconds


ZERO = timedelta(0)
HOUR = timedelta(hours=1)


class ConstantTZInfo(tzinfo):
    """A timezone implementation with a fixed UTC offset."""

    def __init__(self, name, offset):
        self.name = name
        self.offset = offset

    def utcoffset(self, dt):
        return self.offset

    def tzname(self, dt):
        return self.name

    def dst(self, dt):
        return ZERO

    def __repr__(self):
        return self.name


UTC = ConstantTZInfo('UTC', ZERO)
EPOCH_AWARE = datetime.fromtimestamp(0, UTC)
EPOCH = EPOCH_AWARE.replace(tzinfo=None)


def dt_to_timestamp(dt):
    """Convert a datetime to seconds since the Unix epoch."""
    if dt.tzinfo:
        delta = dt - EPOCH_AWARE
    else:
        delta = dt.replace(tzinfo=timezone.utc) - EPOCH_AWARE
    return timedelta.total_seconds(delta)


_NONDIGIT_RE = re.compile(r'\D')


def isoparse(iso_str):
    """Parse the basic ISO-like datetime strings emitted by isoformat()."""
    parts = _NONDIGIT_RE.split(iso_str)
    args = [int(part) for part in parts]
    if len(parts) > 6:
        args[6] = int(parts[6].ljust(6, '0')[:6])
    return datetime(*args)


_BOUNDS = [
    (0, timedelta(seconds=1), 'second'),
    (1, timedelta(seconds=60), 'minute'),
    (1, timedelta(seconds=3600), 'hour'),
    (1, timedelta(days=1), 'day'),
    (1, timedelta(days=7), 'week'),
    (2, timedelta(days=30), 'month'),
    (1, timedelta(days=365), 'year'),
]
_BOUNDS = [(quantity * interval, interval, name)
           for quantity, interval, name in _BOUNDS]
_BOUND_DELTAS = [bound[0] for bound in _BOUNDS]

_FLOAT_PATTERN = r'[+-]?\ *(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
_PARSE_TD_RE = re.compile(
    r'((?P<value>%s)\s*(?P<unit>\w)\w*)' % _FLOAT_PATTERN
)
_PARSE_TD_KW_MAP = {
    unit[0]: unit + 's'
    for _, _, unit in reversed(_BOUNDS[:-2])
}


def parse_timedelta(text):
    """Parse a compact textual timedelta description."""
    kwargs = {}
    for match in _PARSE_TD_RE.finditer(text):
        value = match.group('value')
        unit = match.group('unit')
        try:
            keyword = _PARSE_TD_KW_MAP[unit]
        except KeyError:
            raise ValueError(
                'invalid time unit %r, expected one of %r'
                % (unit, _PARSE_TD_KW_MAP.keys())
            )
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                'invalid time value for unit %r: %r' % (unit, value)
            )
        kwargs[keyword] = value
    return timedelta(**kwargs)


parse_td = parse_timedelta


def _cardinalize_time_unit(unit, value):
    if value == 1:
        return unit
    return unit + 's'


def decimal_relative_time(d, other=None, ndigits=0, cardinalize=True):
    """Return a numeric relative time and its most suitable unit."""
    if other is None:
        if d.tzinfo is None:
            other = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            other = datetime.now(timezone.utc)

    diff = other - d
    seconds = timedelta.total_seconds(diff)
    abs_diff = abs(diff)
    bound_index = bisect.bisect(_BOUND_DELTAS, abs_diff) - 1
    _, unit_delta, unit = _BOUNDS[bound_index]
    value = round(seconds / timedelta.total_seconds(unit_delta), ndigits)

    if cardinalize:
        unit = _cardinalize_time_unit(unit, value)
    return value, unit


def relative_time(d, other=None, ndigits=0):
    """Return a brief human-readable description of relative time."""
    value, unit = decimal_relative_time(d, other, ndigits, cardinalize=True)
    if value == 0:
        return 'just now'
    if value > 0:
        return '%g %s ago' % (value, unit)
    return 'in %g %s' % (-value, unit)


def strpdate(string, fmt):
    """Parse a string into a date using the same formats as strptime()."""
    return date(*time.strptime(string, fmt)[:3])


def _check_date_range_args(start, stop):
    if not isinstance(start, date):
        raise TypeError('expected date object for start, not: %r' % start)
    if not isinstance(stop, date):
        raise TypeError('expected date object for stop, not: %r' % stop)


def daterange(start, stop, step=1, inclusive=False):
    """Generate dates from *start* toward *stop* in day-sized increments."""
    _check_date_range_args(start, stop)

    if not isinstance(step, timedelta):
        step = timedelta(days=step)
    if step == ZERO:
        raise ValueError('step must not be 0 days')

    if step > ZERO:
        comparator = operator.le if inclusive else operator.lt
    else:
        comparator = operator.ge if inclusive else operator.gt

    current = start
    while comparator(current, stop):
        yield current
        current += step


def _add_months(value, amount):
    month_index = value.year * 12 + value.month - 1 + amount
    year, month_index = divmod(month_index, 12)
    return value.replace(year=year, month=month_index + 1)


def monthrange(start, stop, step=1, inclusive=False):
    """Generate the first date of each month in a calendar range."""
    _check_date_range_args(start, stop)
    step = operator.index(step)
    if step == 0:
        raise ValueError('step must not be 0 months')

    start = start.replace(day=1)
    stop = stop.replace(day=1)

    if step > 0:
        comparator = operator.le if inclusive else operator.lt
    else:
        comparator = operator.ge if inclusive else operator.gt

    current = start
    while comparator(current, stop):
        yield current
        current = _add_months(current, step)


def yearrange(start, stop, step=1, inclusive=False):
    """Generate the first date of each year in a calendar range."""
    _check_date_range_args(start, stop)
    step = operator.index(step)
    if step == 0:
        raise ValueError('step must not be 0 years')

    start = start.replace(month=1, day=1)
    stop = stop.replace(month=1, day=1)

    if step > 0:
        comparator = operator.le if inclusive else operator.lt
    else:
        comparator = operator.ge if inclusive else operator.gt

    current = start
    while comparator(current, stop):
        yield current
        current = current.replace(year=current.year + step)


class LocalTZInfo(tzinfo):
    """A tzinfo implementation representing the computer's local timezone."""

    def _isdst(self, dt):
        tt = (
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.weekday(),
            0,
            -1,
        )
        stamp = time.mktime(tt)
        return time.localtime(stamp).tm_isdst > 0

    def utcoffset(self, dt):
        if self._isdst(dt):
            return timedelta(seconds=-time.altzone)
        return timedelta(seconds=-time.timezone)

    def dst(self, dt):
        if self._isdst(dt):
            return timedelta(seconds=time.timezone - time.altzone)
        return ZERO

    def tzname(self, dt):
        if self._isdst(dt):
            return time.tzname[1]
        return time.tzname[0]

    def __repr__(self):
        return 'LocalTZ'


LocalTZ = LocalTZInfo()


def first_sunday_on_or_after(dt):
    """Return the first Sunday equal to or following *dt*."""
    days_to_go = 6 - dt.weekday()
    if days_to_go:
        dt += timedelta(days=days_to_go)
    return dt


DSTSTART_2007 = datetime(1, 3, 8, 2)
DSTEND_2007 = datetime(1, 11, 1, 2)
DSTSTART_1987_2006 = datetime(1, 4, 1, 2)
DSTEND_1987_2006 = datetime(1, 10, 25, 2)
DSTSTART_1967_1986 = datetime(1, 4, 24, 2)
DSTEND_1967_1986 = datetime(1, 10, 25, 2)


class USTimeZone(tzinfo):
    """A simplified timezone implementation for United States timezones."""

    def __init__(self, hours, reprname, stdname, dstname):
        self.stdoffset = timedelta(hours=hours)
        self.reprname = reprname
        self.stdname = stdname
        self.dstname = dstname

    def __repr__(self):
        return self.reprname

    def tzname(self, dt):
        if self.dst(dt) == ZERO:
            return self.stdname
        return self.dstname

    def utcoffset(self, dt):
        return self.stdoffset + self.dst(dt)

    def dst(self, dt):
        if dt is None or dt.tzinfo is not self:
            return ZERO

        if dt.year > 2006:
            start = first_sunday_on_or_after(
                DSTSTART_2007.replace(year=dt.year)
            )
            end = first_sunday_on_or_after(
                DSTEND_2007.replace(year=dt.year)
            )
        elif dt.year > 1986:
            start = first_sunday_on_or_after(
                DSTSTART_1987_2006.replace(year=dt.year)
            )
            end = first_sunday_on_or_after(
                DSTEND_1987_2006.replace(year=dt.year)
            )
        elif dt.year > 1966:
            start = first_sunday_on_or_after(
                DSTSTART_1967_1986.replace(year=dt.year)
            )
            end = first_sunday_on_or_after(
                DSTEND_1967_1986.replace(year=dt.year)
            )
        else:
            return ZERO

        naive_dt = dt.replace(tzinfo=None)
        if start <= naive_dt < end:
            return HOUR
        return ZERO


Eastern = USTimeZone(-5, 'Eastern', 'EST', 'EDT')
Central = USTimeZone(-6, 'Central', 'CST', 'CDT')
Mountain = USTimeZone(-7, 'Mountain', 'MST', 'MDT')
Pacific = USTimeZone(-8, 'Pacific', 'PST', 'PDT')