import re
from datetime import datetime, timedelta, timezone
from datetime import tzinfo as dt_tzinfo
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    SupportsFloat,
    SupportsInt,
    Tuple,
    TypedDict,
    Union,
)

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from arrow import locales
from arrow.constants import DEFAULT_LOCALE
from arrow.util import next_weekday, normalize_timestamp


class ParserError(ValueError):
    pass


class ParserMatchError(ParserError):
    pass


_WEEKDATE_ELEMENT = Union[str, bytes, SupportsInt, bytearray]

_FORMAT_TYPE = Literal[
    "YYYY", "YY", "MM", "M", "DDDD", "DDD", "DD", "D", "HH", "H",
    "hh", "h", "mm", "m", "ss", "s", "X", "x", "ZZZ", "ZZ", "Z",
    "S", "W", "MMMM", "MMM", "Do", "dddd", "ddd", "d", "a", "A",
]


class _Parts(TypedDict, total=False):
    year: int
    month: int
    day_of_year: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    timestamp: float
    expanded_timestamp: int
    tzinfo: dt_tzinfo
    am_pm: Literal["am", "pm"]
    day_of_week: int
    weekdate: Tuple[_WEEKDATE_ELEMENT, _WEEKDATE_ELEMENT, Optional[_WEEKDATE_ELEMENT]]


class TzinfoParser:
    _TZINFO_RE: ClassVar[Pattern[str]] = re.compile(
        r"^(?:(?P<local>local)|(?P<utc>utc|UTC|Z)|"
        r"(?P<offset>[+-]\d{2}(?::?\d{2})?)|"
        r"(?P<name>\w[\w+\-/]+))$"
    )

    @classmethod
    def parse(cls, tzinfo_string: str) -> dt_tzinfo:
        if not isinstance(tzinfo_string, str):
            raise ParserError(
                "Could not parse timezone expression {!r}".format(tzinfo_string)
            )

        match = cls._TZINFO_RE.match(tzinfo_string)
        if match is None:
            raise ParserError(
                "Could not parse timezone expression {!r}".format(tzinfo_string)
            )

        if match.group("local") is not None:
            local_tz = datetime.now().astimezone().tzinfo
            return timezone.utc if local_tz is None else local_tz

        if match.group("utc") is not None:
            return timezone.utc

        offset = match.group("offset")
        if offset is not None:
            sign = 1 if offset.startswith("+") else -1
            values = offset[1:].replace(":", "")
            hours = int(values[:2])
            minutes = int(values[2:]) if len(values) > 2 else 0
            try:
                return timezone(sign * timedelta(hours=hours, minutes=minutes))
            except ValueError as exc:
                raise ParserError(
                    "Could not parse timezone expression {!r}".format(tzinfo_string)
                ) from exc

        name = match.group("name")
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ParserError(
                "Could not parse timezone expression {!r}".format(tzinfo_string)
            ) from exc


class DateTimeParser:
    _FORMAT_RE: ClassVar[Pattern[str]] = re.compile(
        r"(YYY?Y?|MM?M?M?|Do|DD?D?D?|d?d?d?d|HH?|hh?|mm?|ss?|"
        r"S+|ZZ?Z?|a|A|x|X|W)"
    )
    _ESCAPE_RE: ClassVar[Pattern[str]] = re.compile(r"\[[^\[\]]*\]")

    _ONE_OR_TWO_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d{1,2}")
    _ONE_OR_TWO_OR_THREE_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d{1,3}")
    _ONE_OR_MORE_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d+")
    _TWO_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d{2}")
    _THREE_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d{3}")
    _FOUR_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d{4}")
    _TZ_Z_RE: ClassVar[Pattern[str]] = re.compile(r"([\+\-])(\d{2})(?:(\d{2}))?|Z")
    _TZ_ZZ_RE: ClassVar[Pattern[str]] = re.compile(r"([\+\-])(\d{2})(?:\:(\d{2}))?|Z")
    _TZ_NAME_RE: ClassVar[Pattern[str]] = re.compile(r"\w[\w+\-/]+")
    _TIMESTAMP_RE: ClassVar[Pattern[str]] = re.compile(r"^\-?\d+\.?\d+$")
    _TIMESTAMP_EXPANDED_RE: ClassVar[Pattern[str]] = re.compile(r"^\-?\d+$")
    _TIME_RE: ClassVar[Pattern[str]] = re.compile(
        r"^(\d{2})(?:\:?(\d{2}))?(?:\:?(\d{2}))?(?:([\.,])(\d+))?$"
    )
    _WEEK_DATE_RE: ClassVar[Pattern[str]] = re.compile(
        r"(?P<year>\d{4})[\-]?W(?P<week>\d{2})[\-]?(?P<day>\d)?"
    )

    _BASE_INPUT_RE_MAP: ClassVar[Dict[_FORMAT_TYPE, Pattern[str]]] = {
        "YYYY": _FOUR_DIGIT_RE,
        "YY": _TWO_DIGIT_RE,
        "MM": _TWO_DIGIT_RE,
        "M": _ONE_OR_TWO_DIGIT_RE,
        "DDDD": _THREE_DIGIT_RE,
        "DDD": _ONE_OR_TWO_OR_THREE_DIGIT_RE,
        "DD": _TWO_DIGIT_RE,
        "D": _ONE_OR_TWO_DIGIT_RE,
        "HH": _TWO_DIGIT_RE,
        "H": _ONE_OR_TWO_DIGIT_RE,
        "hh": _TWO_DIGIT_RE,
        "h": _ONE_OR_TWO_DIGIT_RE,
        "mm": _TWO_DIGIT_RE,
        "m": _ONE_OR_TWO_DIGIT_RE,
        "ss": _TWO_DIGIT_RE,
        "s": _ONE_OR_TWO_DIGIT_RE,
        "X": _TIMESTAMP_RE,
        "x": _TIMESTAMP_EXPANDED_RE,
        "ZZZ": _TZ_NAME_RE,
        "ZZ": _TZ_ZZ_RE,
        "Z": _TZ_Z_RE,
        "S": _ONE_OR_MORE_DIGIT_RE,
        "W": _WEEK_DATE_RE,
    }

    SEPARATORS: ClassVar[List[str]] = ["-", "/", "."]

    def __init__(self, locale: str = DEFAULT_LOCALE, cache_size: int = 0) -> None:
        self.locale = locales.get_locale(locale)
        self._input_re_map = self._BASE_INPUT_RE_MAP.copy()
        self._input_re_map.update(
            {
                "MMMM": self._generate_choice_re(
                    self.locale.month_names[1:], re.IGNORECASE
                ),
                "MMM": self._generate_choice_re(
                    self.locale.month_abbreviations[1:], re.IGNORECASE
                ),
                "Do": re.compile(self.locale.ordinal_day_re),
                "dddd": self._generate_choice_re(
                    self.locale.day_names[1:], re.IGNORECASE
                ),
                "ddd": self._generate_choice_re(
                    self.locale.day_abbreviations[1:], re.IGNORECASE
                ),
                "d": re.compile(r"[1-7]"),
                "a": self._generate_choice_re(
                    (self.locale.meridians["am"], self.locale.meridians["pm"])
                ),
                "A": self._generate_choice_re(self.locale.meridians.values()),
            }
        )
        if cache_size > 0:
            self._generate_pattern_re = lru_cache(maxsize=cache_size)(
                self._generate_pattern_re
            )

    @staticmethod
    def _generate_choice_re(
        choices: Iterable[str], flags: int = 0
    ) -> Pattern[str]:
        values = [str(choice) for choice in choices if choice is not None]
        values.sort(key=len, reverse=True)
        return re.compile("(?:{})".format("|".join(re.escape(x) for x in values)), flags)

    def _format_tokens(self, fmt: str) -> List[Tuple[Optional[str], str]]:
        result: List[Tuple[Optional[str], str]] = []
        position = 0
        for match in self._FORMAT_RE.finditer(fmt):
            if match.start() > position:
                literal = fmt[position:match.start()]
                result.extend(self._literal_tokens(literal))
            result.append((match.group(0), match.group(0)))
            position = match.end()
        if position < len(fmt):
            result.extend(self._literal_tokens(fmt[position:]))
        return result

    @staticmethod
    def _literal_tokens(value: str) -> List[Tuple[Optional[str], str]]:
        pieces: List[Tuple[Optional[str], str]] = []
        pos = 0
        for match in re.finditer(r"\[([^\[\]]*)\]", value):
            if match.start() > pos:
                pieces.append((None, value[pos:match.start()]))
            pieces.append((None, match.group(1)))
            pos = match.end()
        if pos < len(value):
            pieces.append((None, value[pos:]))
        return pieces

    def _generate_pattern_re(self, fmt: str) -> Pattern[str]:
        expressions: List[str] = []
        for token, value in self._format_tokens(fmt):
            if token is None:
                expressions.append(re.escape(value))
            else:
                expressions.append("({})".format(self._input_re_map[token].pattern))
        return re.compile("^{}$".format("".join(expressions)), re.IGNORECASE)

    def parse_iso(
        self, datetime_string: str, normalize_whitespace: bool = False
    ) -> datetime:
        if normalize_whitespace:
            datetime_string = re.sub(r"\s+", " ", datetime_string.strip())

        text = datetime_string
        date_text = text
        time_text: Optional[str] = None

        split = re.match(r"^(.+?)(?:T| )(.+)$", text)
        if split is not None:
            date_text, time_text = split.group(1), split.group(2)

        parts: _Parts = {}

        week = re.fullmatch(r"(\d{4})-?W(\d{2})(?:-?(\d))?", date_text)
        ordinal = re.fullmatch(r"(\d{4})-?(\d{3})", date_text)
        calendar = re.fullmatch(
            r"(\d{4})(?:-?(\d{1,2})(?:-?(\d{1,2}))?)?", date_text
        )

        if week is not None:
            parts["weekdate"] = (week.group(1), week.group(2), week.group(3))
        elif ordinal is not None:
            parts["year"] = int(ordinal.group(1))
            parts["day_of_year"] = int(ordinal.group(2))
        elif calendar is not None:
            parts["year"] = int(calendar.group(1))
            if calendar.group(2) is not None:
                parts["month"] = int(calendar.group(2))
            if calendar.group(3) is not None:
                parts["day"] = int(calendar.group(3))
        else:
            raise ParserError("Could not parse {!r} as ISO 8601".format(datetime_string))

        if time_text is not None:
            tz_match = re.search(r"(Z|[+-]\d{2}(?::?\d{2})?)$", time_text)
            if tz_match is not None:
                tz_value = tz_match.group(1)
                try:
                    parts["tzinfo"] = TzinfoParser.parse(tz_value)
                except ParserError as exc:
                    raise ParserError(
                        "Could not parse {!r} as ISO 8601".format(datetime_string)
                    ) from exc
                time_text = time_text[:tz_match.start()]

            time_match = self._TIME_RE.fullmatch(time_text)
            if time_match is None:
                raise ParserError(
                    "Could not parse {!r} as ISO 8601".format(datetime_string)
                )

            parts["hour"] = int(time_match.group(1))
            if time_match.group(2) is not None:
                parts["minute"] = int(time_match.group(2))
            if time_match.group(3) is not None:
                parts["second"] = int(time_match.group(3))
            if time_match.group(5) is not None:
                fraction = time_match.group(5)
                parts["microsecond"] = int((fraction + "000000")[:6])

        try:
            return self._build_datetime(parts)
        except (TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ParserError):
                raise
            raise ParserError(
                "Could not parse {!r} as ISO 8601".format(datetime_string)
            ) from exc

    def parse(self, datetime_string: str, fmt: str) -> datetime:
        if not isinstance(datetime_string, str):
            raise ParserError("Expected a string, got {!r}".format(datetime_string))
        if not isinstance(fmt, str):
            raise ParserError("Expected a format string, got {!r}".format(fmt))
        return self._parse(datetime_string, fmt)

    def parse_multiformat(
        self, datetime_string: str, formats: Iterable[str]
    ) -> datetime:
        return self._parse_multiformat(datetime_string, formats)

    def _parse_multiformat(
        self, datetime_string: str, formats: Iterable[str]
    ) -> datetime:
        format_list = list(formats)
        for fmt in format_list:
            try:
                return self._parse(datetime_string, fmt)
            except ParserMatchError:
                continue
        raise ParserError(
            "Could not match input {!r} to any of the following formats: {}".format(
                datetime_string, ", ".join(format_list)
            )
        )

    def _parse(self, datetime_string: str, fmt: str) -> datetime:
        tokens = self._format_tokens(fmt)
        pattern = self._generate_pattern_re(fmt)
        match = pattern.match(datetime_string)
        if match is None:
            raise ParserMatchError(
                "Failed to match {!r} when parsing with format {!r}".format(
                    datetime_string, fmt
                )
            )

        parts: _Parts = {}
        values = iter(match.groups())
        try:
            for token, _ in tokens:
                if token is not None:
                    self._parse_token(token, next(values), parts)
            return self._build_datetime(parts)
        except ParserMatchError:
            raise
        except ParserError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ParserMatchError(
                "Failed to match {!r} when parsing with format {!r}".format(
                    datetime_string, fmt
                )
            ) from exc

    def _parse_token(self, token: str, value: str, parts: _Parts) -> None:
        if token == "YYYY":
            parts["year"] = int(value)
        elif token == "YY":
            year = int(value)
            parts["year"] = 1900 + year if year > 68 else 2000 + year
        elif token in ("MM", "M"):
            parts["month"] = int(value)
        elif token == "MMMM" or token == "MMM":
            parts["month"] = self.locale.month_number(value)
        elif token in ("DDDD", "DDD"):
            parts["day_of_year"] = int(value)
        elif token in ("DD", "D"):
            parts["day"] = int(value)
        elif token == "Do":
            parts["day"] = self.locale.ordinal_number(value)
        elif token in ("HH", "H", "hh", "h"):
            parts["hour"] = int(value)
        elif token in ("mm", "m"):
            parts["minute"] = int(value)
        elif token in ("ss", "s"):
            parts["second"] = int(value)
        elif token == "S":
            parts["microsecond"] = int((value + "000000")[:6])
        elif token == "X":
            parts["timestamp"] = float(value)
        elif token == "x":
            parts["expanded_timestamp"] = int(value)
        elif token in ("Z", "ZZ", "ZZZ"):
            parts["tzinfo"] = TzinfoParser.parse(value)
        elif token in ("a", "A"):
            meridian = self.locale.meridian(value)
            parts["am_pm"] = "pm" if meridian == "pm" else "am"
        elif token in ("dddd", "ddd"):
            parts["day_of_week"] = self.locale.day_number(value)
        elif token == "d":
            parts["day_of_week"] = int(value)
        elif token == "W":
            match = self._WEEK_DATE_RE.fullmatch(value)
            if match is None:
                raise ParserMatchError("Invalid ISO week date {!r}".format(value))
            parts["weekdate"] = (
                match.group("year"),
                match.group("week"),
                match.group("day"),
            )

    def _build_datetime(self, parts: _Parts) -> datetime:
        tzinfo = parts.get("tzinfo")

        if "timestamp" in parts:
            return datetime.fromtimestamp(
                normalize_timestamp(parts["timestamp"]), tz=tzinfo
            )

        if "expanded_timestamp" in parts:
            timestamp = normalize_timestamp(parts["expanded_timestamp"] / 1000000.0)
            return datetime.fromtimestamp(timestamp, tz=tzinfo)

        if "weekdate" in parts:
            year_value, week_value, weekday_value = parts["weekdate"]
            year = int(year_value)
            week = int(week_value)
            weekday = int(weekday_value) if weekday_value is not None else 1
            try:
                result = datetime.fromisocalendar(year, week, weekday)
            except ValueError as exc:
                raise ParserError(
                    "Could not parse ISO week date {!r}".format(parts["weekdate"])
                ) from exc
            if tzinfo is not None:
                result = result.replace(tzinfo=tzinfo)
            return result

        year = parts.get("year", 1)
        month = parts.get("month", 1)
        day = parts.get("day", 1)

        if "day_of_year" in parts:
            ordinal = parts["day_of_year"]
            try:
                first = datetime(year, 1, 1)
                candidate = first + timedelta(days=ordinal - 1)
            except (ValueError, OverflowError) as exc:
                raise ParserError(
                    "The provided day of year {} is invalid for year {}".format(
                        ordinal, year
                    )
                ) from exc
            if candidate.year != year:
                raise ParserError(
                    "The provided day of year {} is invalid for year {}".format(
                        ordinal, year
                    )
                )
            month, day = candidate.month, candidate.day

        hour = parts.get("hour", 0)
        minute = parts.get("minute", 0)
        second = parts.get("second", 0)
        microsecond = parts.get("microsecond", 0)

        am_pm = parts.get("am_pm")
        if am_pm is not None:
            if hour > 12:
                raise ParserError("Hour must be between 0 and 12 when AM/PM is specified")
            if am_pm == "pm" and hour < 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0

        result = datetime(year, month, day, hour, minute, second, microsecond, tzinfo)

        if "day_of_week" in parts:
            weekday = parts["day_of_week"]
            if weekday >= 1 and weekday <= 7:
                weekday -= 1
            result = next_weekday(result, weekday)

        return result