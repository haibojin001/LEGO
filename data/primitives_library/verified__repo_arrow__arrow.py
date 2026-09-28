import calendar
import re
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from datetime import tzinfo as dt_tzinfo
from math import trunc
from time import struct_time

from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta

from arrow import formatter, locales, parser, util
from arrow.constants import DEFAULT_LOCALE, DEHUMANIZE_LOCALES

TZ_EXPR = object

_GRANULARITIES = (
    "second", "minute", "hour", "day", "week", "month", "quarter", "year"
)


class Arrow:
    resolution = dt_datetime.resolution

    _ATTRS = ["year", "month", "day", "hour", "minute", "second", "microsecond"]
    _ATTRS_PLURAL = [x + "s" for x in _ATTRS]
    _MONTHS_PER_QUARTER = 3
    _MONTHS_PER_YEAR = 12
    _SECS_PER_MINUTE = 60
    _SECS_PER_HOUR = 3600
    _SECS_PER_DAY = 86400
    _SECS_PER_WEEK = 604800
    _SECS_PER_MONTH = 2635200.0
    _SECS_PER_QUARTER = 7905600.0
    _SECS_PER_YEAR = 31536000
    _SECS_MAP = {
        "second": 1.0,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
        "week": 604800,
        "month": 2635200.0,
        "quarter": 7905600.0,
        "year": 31536000,
    }

    def __init__(self, year, month, day, hour=0, minute=0, second=0,
                 microsecond=0, tzinfo=None, **kwargs):
        if tzinfo is None:
            tzinfo = timezone.utc
        elif (isinstance(tzinfo, dt_tzinfo) and hasattr(tzinfo, "localize")
              and hasattr(tzinfo, "zone") and tzinfo.zone):
            tzinfo = parser.TzinfoParser.parse(tzinfo.zone)
        elif isinstance(tzinfo, str):
            tzinfo = parser.TzinfoParser.parse(tzinfo)
        self._datetime = dt_datetime(
            year, month, day, hour, minute, second, microsecond, tzinfo,
            fold=kwargs.get("fold", 0)
        )

    @classmethod
    def now(cls, tzinfo=None):
        if tzinfo is None:
            tzinfo = dt_datetime.now().astimezone().tzinfo
        elif isinstance(tzinfo, str):
            tzinfo = parser.TzinfoParser.parse(tzinfo)
        return cls.fromdatetime(dt_datetime.now(tzinfo))

    @classmethod
    def utcnow(cls):
        return cls.fromdatetime(dt_datetime.now(timezone.utc))

    @classmethod
    def fromtimestamp(cls, timestamp, tzinfo=None):
        if tzinfo is None:
            tzinfo = dt_datetime.now().astimezone().tzinfo
        elif isinstance(tzinfo, str):
            tzinfo = parser.TzinfoParser.parse(tzinfo)
        if not util.is_timestamp(timestamp):
            raise ValueError("The provided timestamp {!r} is invalid.".format(timestamp))
        return cls.fromdatetime(
            dt_datetime.fromtimestamp(util.normalize_timestamp(float(timestamp)), tzinfo)
        )

    @classmethod
    def utcfromtimestamp(cls, timestamp):
        if not util.is_timestamp(timestamp):
            raise ValueError("The provided timestamp {!r} is invalid.".format(timestamp))
        return cls.fromdatetime(
            dt_datetime.fromtimestamp(util.normalize_timestamp(float(timestamp)), timezone.utc)
        )

    @classmethod
    def fromdatetime(cls, dt, tzinfo=None):
        if tzinfo is None:
            tzinfo = dt.tzinfo if dt.tzinfo is not None else timezone.utc
        return cls(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                   dt.microsecond, tzinfo, fold=getattr(dt, "fold", 0))

    @classmethod
    def fromdate(cls, date, tzinfo=None):
        return cls(date.year, date.month, date.day, tzinfo=tzinfo)

    @classmethod
    def strptime(cls, date_str, fmt, tzinfo=None):
        dt = dt_datetime.strptime(date_str, fmt)
        return cls.fromdatetime(dt, tzinfo)

    @classmethod
    def fromisoformat(cls, date_string):
        return cls.fromdatetime(dt_datetime.fromisoformat(date_string))

    @classmethod
    def fromordinal(cls, ordinal):
        return cls.fromdatetime(dt_datetime.fromordinal(ordinal))

    @classmethod
    def fromarrows(cls, arrow):
        if isinstance(arrow, cls):
            return arrow
        return cls.fromdatetime(arrow.datetime)

    @classmethod
    def range(cls, frame, start, end=None, tz=None, limit=None):
        if frame not in cls._ATTRS_PLURAL:
            raise ValueError("Invalid frame {}".format(frame))
        if isinstance(start, dt_datetime):
            start = cls.fromdatetime(start)
        if end is not None and isinstance(end, dt_datetime):
            end = cls.fromdatetime(end)
        if tz is not None:
            start = start.to(tz)
            if end is not None:
                end = end.to(tz)
        if end is None:
            end = cls.max
        current = start
        count = 0
        while current <= end and (limit is None or count < limit):
            yield current
            current = current.shift(**{frame: 1})
            count += 1

    @classmethod
    def span_range(cls, frame, start, end, tz=None, limit=None, bounds="[)",
                   exact=False):
        if frame not in _GRANULARITIES:
            raise ValueError("Invalid frame {}".format(frame))
        if isinstance(start, dt_datetime):
            start = cls.fromdatetime(start)
        if isinstance(end, dt_datetime):
            end = cls.fromdatetime(end)
        if tz is not None:
            start, end = start.to(tz), end.to(tz)
        if exact:
            current = start
        else:
            current = start.floor(frame)
        count = 0
        while current <= end and (limit is None or count < limit):
            if exact:
                span_end = current.shift(**{frame + "s": 1}).shift(microseconds=-1)
                if span_end > end:
                    span_end = end
                yield current, span_end
                current = span_end.shift(microseconds=1)
            else:
                yield current.span(frame, bounds=bounds)
                current = current.shift(**{frame + "s": 1})
            count += 1

    @classmethod
    def interval(cls, frame, start, end, interval=1, tz=None, bounds="[)",
                 exact=False):
        if interval < 1:
            raise ValueError("Interval must be greater than zero.")
        if isinstance(start, dt_datetime):
            start = cls.fromdatetime(start)
        if isinstance(end, dt_datetime):
            end = cls.fromdatetime(end)
        if tz is not None:
            start, end = start.to(tz), end.to(tz)
        current = start if exact else start.floor(frame)
        while current <= end:
            nxt = current.shift(**{frame + "s": interval})
            finish = nxt.shift(microseconds=-1)
            if exact and finish > end:
                finish = end
            yield current, finish
            current = nxt

    def clone(self):
        return self.__class__.fromdatetime(self._datetime)

    def replace(self, **kwargs):
        absolute = {}
        tzinfo = kwargs.pop("tzinfo", self.tzinfo)
        for key, value in kwargs.items():
            if key in self._ATTRS:
                absolute[key] = value
            elif key in self._ATTRS_PLURAL:
                raise AttributeError("Cannot replace plural attribute {!r}".format(key))
            elif key == "fold":
                absolute[key] = value
            else:
                raise AttributeError("Unknown attribute {!r}".format(key))
        dt = self._datetime.replace(tzinfo=tzinfo, **absolute)
        return self.__class__.fromdatetime(dt)

    def shift(self, check_imaginary=True, **kwargs):
        relative = {}
        for key, value in kwargs.items():
            if key in self._ATTRS_PLURAL or key in ("weeks", "quarters"):
                relative[key] = value
            elif key in self._ATTRS:
                raise AttributeError("Cannot shift by singular attribute {!r}".format(key))
            else:
                raise AttributeError("Unknown attribute {!r}".format(key))
        if "quarters" in relative:
            relative["months"] = relative.get("months", 0) + relative.pop("quarters") * 3
        dt = self._datetime + relativedelta(**relative)
        if check_imaginary and not dateutil_tz.datetime_exists(dt):
            dt = dateutil_tz.resolve_imaginary(dt)
        return self.__class__.fromdatetime(dt)

    def to(self, tz):
        if isinstance(tz, str):
            tz = parser.TzinfoParser.parse(tz)
        return self.__class__.fromdatetime(self._datetime.astimezone(tz))

    def span(self, frame, count=1, bounds="[)", exact=False, week_start=1):
        if frame not in _GRANULARITIES:
            raise ValueError("Invalid frame {}".format(frame))
        if bounds not in ("[)", "()", "(]", "[]"):
            raise ValueError("Invalid bounds {}".format(bounds))
        if count < 1:
            raise ValueError("Count must be greater than zero.")
        if exact:
            start = self
        else:
            start = self.floor(frame, week_start=week_start)
        end = start.shift(**{frame + "s": count})
        if bounds[0] == "(":
            start = start.shift(microseconds=1)
        if bounds[1] == ")":
            end = end.shift(microseconds=-1)
        return start, end

    def floor(self, frame, **kwargs):
        return self.span(frame, **kwargs)[0]

    def ceil(self, frame, **kwargs):
        return self.span(frame, **kwargs)[1]

    def _floor_datetime(self, frame, week_start=1):
        dt = self._datetime
        if frame == "year":
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if frame == "quarter":
            month = ((dt.month - 1) // 3) * 3 + 1
            return dt.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        if frame == "month":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if frame == "week":
            if week_start < 1 or week_start > 7:
                raise ValueError("week_start must be between 1 and 7.")
            delta = (dt.isoweekday() - week_start) % 7
            return (dt - timedelta(days=delta)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if frame == "day":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if frame == "hour":
            return dt.replace(minute=0, second=0, microsecond=0)
        if frame == "minute":
            return dt.replace(second=0, microsecond=0)
        if frame == "second":
            return dt.replace(microsecond=0)
        raise ValueError("Invalid frame {}".format(frame))

    def format(self, fmt="YYYY-MM-DD HH:mm:ssZZ", locale=DEFAULT_LOCALE):
        return formatter.DateTimeFormatter(locale).format(self._datetime, fmt)

    def humanize(self, other=None, locale=DEFAULT_LOCALE, only_distance=False,
                 granularity="auto"):
        locale_obj = locales.get_locale(locale)
        if other is None:
            other = self.__class__.now(self.tzinfo)
        elif isinstance(other, dt_datetime):
            other = self.__class__.fromdatetime(other)
        elif not isinstance(other, Arrow):
            raise TypeError("Invalid other argument.")
        delta = self._datetime - other._datetime
        seconds = delta.total_seconds()
        if isinstance(granularity, str):
            granularities = [granularity]
        else:
            granularities = list(granularity)
        if granularities == ["auto"]:
            if abs(seconds) < 10:
                return locale_obj.describe("now", only_distance=only_distance)
            frame = "second"
            for name in ("year", "month", "week", "day", "hour", "minute", "second"):
                if abs(seconds) >= self._SECS_MAP[name]:
                    frame = name
                    break
            value = trunc(abs(seconds) / self._SECS_MAP[frame])
            if value == 0:
                value = 1
            return locale_obj.describe(
                frame if value == 1 else frame + "s",
                value if seconds >= 0 else -value,
                only_distance=only_distance,
            )
        pieces = []
        remainder = abs(seconds)
        for frame in granularities:
            if frame not in self._SECS_MAP:
                raise ValueError("Invalid granularity {}".format(frame))
            amount = trunc(remainder / self._SECS_MAP[frame])
            remainder -= amount * self._SECS_MAP[frame]
            if amount:
                pieces.append(locale_obj.describe(
                    frame if amount == 1 else frame + "s", amount, only_distance=True
                ))
        if not pieces:
            pieces = [locale_obj.describe("now", only_distance=True)]
        text = locale_obj.and_join(pieces) if len(pieces) > 1 else pieces[0]
        if only_distance:
            return text
        return locale_obj.relative(text, "seconds", 1 if seconds >= 0 else -1)

    def dehumanize(self, input_string, locale=DEFAULT_LOCALE):
        if locale not in DEHUMANIZE_LOCALES:
            raise ValueError("Dehumanize does not support locale {!r}".format(locale))
        locale_obj = locales.get_locale(locale)
        text = input_string.lower()
        direction = 1
        for token in getattr(locale_obj, "past", []):
            if token.lower() in text:
                direction = -1
        for token in getattr(locale_obj, "future", []):
            if token.lower() in text:
                direction = 1
        units = {
            "year": "years", "month": "months", "week": "weeks", "day": "days",
            "hour": "hours", "minute": "minutes", "second": "seconds",
        }
        values = {}
        for singular, plural in units.items():
            pattern = r"(-?\d+)\s+" + re.escape(singular) + r"s?"
            found = re.search(pattern, text)
            if found:
                values[plural] = int(found.group(1)) * direction
        if not values:
            raise ValueError("Invalid input string")
        return self.shift(**values)

    @property
    def datetime(self):
        return self._datetime

    @property
    def naive(self):
        return self._datetime.replace(tzinfo=None)

    @property
    def date(self):
        return self._datetime.date()

    @property
    def time(self):
        return self._datetime.time()

    @property
    def timetz(self):
        return self._datetime.timetz()

    @property
    def tzinfo(self):
        return self._datetime.tzinfo

    @property
    def timestamp(self):
        return self._datetime.timestamp()

    @property
    def int_timestamp(self):
        return int(self._datetime.timestamp())

    @property
    def float_timestamp(self):
        return self._datetime.timestamp()

    @property
    def fold(self):
        return self._datetime.fold

    @property
    def ambiguous(self):
        return dateutil_tz.datetime_ambiguous(self._datetime)

    @property
    def imaginary(self):
        return not dateutil_tz.datetime_exists(self._datetime)

    def isoformat(self, sep="T", timespec="auto"):
        return self._datetime.isoformat(sep, timespec)

    def for_json(self):
        return self.isoformat()

    def isocalendar(self):
        return self._datetime.isocalendar()

    def weekday(self):
        return self._datetime.weekday()

    def isoweekday(self):
        return self._datetime.isoweekday()

    def toordinal(self):
        return self._datetime.toordinal()

    def tzname(self):
        return self._datetime.tzname()

    def utcoffset(self):
        return self._datetime.utcoffset()

    def dst(self):
        return self._datetime.dst()

    def timetuple(self):
        return self._datetime.timetuple()

    def utctimetuple(self):
        return self._datetime.utctimetuple()

    def __repr__(self):
        return "<{} [{}]>".format(self.__class__.__name__, self.isoformat())

    def __str__(self):
        return self.isoformat()

    def __format__(self, formatstr):
        return self.format(formatstr) if formatstr else str(self)

    def __getattr__(self, name):
        if name in self._ATTRS:
            return getattr(self._datetime, name)
        raise AttributeError(name)

    def __hash__(self):
        return hash(self._datetime)

    def __eq__(self, other):
        if isinstance(other, Arrow):
            return self._datetime == other._datetime
        return self._datetime == other

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self._datetime < (other._datetime if isinstance(other, Arrow) else other)

    def __le__(self, other):
        return self._datetime <= (other._datetime if isinstance(other, Arrow) else other)

    def __gt__(self, other):
        return self._datetime > (other._datetime if isinstance(other, Arrow) else other)

    def __ge__(self, other):
        return self._datetime >= (other._datetime if isinstance(other, Arrow) else other)

    def __add__(self, other):
        if isinstance(other, (timedelta, relativedelta)):
            return self.__class__.fromdatetime(self._datetime + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (timedelta, relativedelta)):
            return self.__class__.fromdatetime(self._datetime - other)
        if isinstance(other, Arrow):
            return self._datetime - other._datetime
        return self._datetime - other

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, dt_datetime):
            return other - self._datetime
        return NotImplemented


Arrow.min = Arrow.fromdatetime(dt_datetime.min.replace(tzinfo=timezone.utc))
Arrow.max = Arrow.fromdatetime(dt_datetime.max.replace(tzinfo=timezone.utc))