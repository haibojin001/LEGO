from __future__ import annotations

import datetime
import re

from re import Match
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import cast

import pendulum

from pendulum.locales.locale import Locale

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from pendulum import Timezone


_MATCH_1 = r"\d"
_MATCH_2 = r"\d\d"
_MATCH_3 = r"\d{3}"
_MATCH_4 = r"\d{4}"
_MATCH_6 = r"[+-]?\d{6}"
_MATCH_1_TO_2 = r"\d\d?"
_MATCH_1_TO_2_LEFT_PAD = r"[0-9 ]\d?"
_MATCH_1_TO_3 = r"\d{1,3}"
_MATCH_1_TO_4 = r"\d{1,4}"
_MATCH_1_TO_6 = r"[+-]?\d{1,6}"
_MATCH_3_TO_4 = r"\d{3}\d?"
_MATCH_5_TO_6 = r"\d{5}\d?"
_MATCH_UNSIGNED = r"\d+"
_MATCH_SIGNED = r"[+-]?\d+"
_MATCH_OFFSET = r"[Zz]|[+-]\d\d:?\d\d"
_MATCH_SHORT_OFFSET = r"[Zz]|[+-]\d\d(?::?\d\d)?"
_MATCH_TIMESTAMP = r"[+-]?\d+(\.\d{1,6})?"
_MATCH_WORD = (
    "(?i)[0-9]*"
    "['a-z\u00a0-\u05ff\u0700-\ud7ff\uf900-\ufdcf\ufdf0-\uffef]+"
    r"|[\u0600-\u06FF/]+(\s*?[\u0600-\u06FF]+){1,2}"
)
_MATCH_TIMEZONE = "[A-Za-z0-9-+]+(/[A-Za-z0-9-+_]+)?"


class Formatter:
    _TOKENS: str = (
        r"\[([^\[]*)\]|\\(.)|"
        "("
        "Mo|MM?M?M?"
        "|Do|DDDo|DD?D?D?|ddd?d?|do?|eo?"
        "|E{1,4}"
        "|w[o|w]?|W[o|W]?|Qo?"
        "|YYYY|YY|Y"
        "|gg(ggg?)?|GG(GGG?)?"
        "|a|A"
        "|hh?|HH?|kk?"
        "|mm?|ss?|S{1,9}"
        "|x|X"
        "|zz?|ZZ?"
        "|LTS|LT|LL?L?L?"
        ")"
    )

    _FORMAT_RE: re.Pattern[str] = re.compile(_TOKENS)
    _FROM_FORMAT_RE: re.Pattern[str] = re.compile(r"(?<!\\\[)" + _TOKENS + r"(?!\\\])")

    _LOCALIZABLE_TOKENS: ClassVar[
        dict[str, str | Callable[[Locale], Sequence[str]] | None]
    ] = {
        "Qo": None,
        "MMMM": "months.wide",
        "MMM": "months.abbreviated",
        "Mo": None,
        "DDDo": None,
        "Do": lambda locale: tuple(
            rf"\d+{ordinal}" for ordinal in locale.get("custom.ordinal").values()
        ),
        "dddd": "days.wide",
        "ddd": "days.abbreviated",
        "dd": "days.short",
        "do": None,
        "e": None,
        "eo": None,
        "Wo": None,
        "wo": None,
        "A": lambda locale: (
            locale.translation("day_periods.am"),
            locale.translation("day_periods.pm"),
        ),
        "a": lambda locale: (
            locale.translation("day_periods.am").lower(),
            locale.translation("day_periods.pm").lower(),
        ),
    }

    _TOKENS_RULES: ClassVar[dict[str, Callable[[pendulum.DateTime], str]]] = {
        "YYYY": lambda dt: f"{dt.year:d}",
        "YY": lambda dt: f"{dt.year:d}"[2:],
        "Y": lambda dt: f"{dt.year:d}",
        "Q": lambda dt: f"{dt.quarter:d}",
        "MM": lambda dt: f"{dt.month:02d}",
        "M": lambda dt: f"{dt.month:d}",
        "DD": lambda dt: f"{dt.day:02d}",
        "D": lambda dt: f"{dt.day:d}",
        "DDDD": lambda dt: f"{dt.day_of_year:03d}",
        "DDD": lambda dt: f"{dt.day_of_year:d}",
        "d": lambda dt: f"{(dt.day_of_week + 1) % 7:d}",
        "E": lambda dt: f"{dt.isoweekday():d}",
        "HH": lambda dt: f"{dt.hour:02d}",
        "H": lambda dt: f"{dt.hour:d}",
        "hh": lambda dt: f"{dt.hour % 12 or 12:02d}",
        "h": lambda dt: f"{dt.hour % 12 or 12:d}",
        "kk": lambda dt: f"{dt.hour or 24:02d}",
        "k": lambda dt: f"{dt.hour or 24:d}",
        "mm": lambda dt: f"{dt.minute:02d}",
        "m": lambda dt: f"{dt.minute:d}",
        "ss": lambda dt: f"{dt.second:02d}",
        "s": lambda dt: f"{dt.second:d}",
        "S": lambda dt: f"{dt.microsecond // 100000:01d}",
        "SS": lambda dt: f"{dt.microsecond // 10000:02d}",
        "SSS": lambda dt: f"{dt.microsecond // 1000:03d}",
        "SSSS": lambda dt: f"{dt.microsecond // 100:04d}",
        "SSSSS": lambda dt: f"{dt.microsecond // 10:05d}",
        "SSSSSS": lambda dt: f"{dt.microsecond:06d}",
        "X": lambda dt: f"{dt.int_timestamp:d}",
        "x": lambda dt: f"{dt.int_timestamp * 1000 + dt.microsecond // 1000:d}",
        "zz": lambda dt: f"{dt.tzname() if dt.tzinfo is not None else ''}",
        "z": lambda dt: f"{dt.timezone_name or ''}",
    }

    _DATE_FORMATS: ClassVar[dict[str, str]] = {
        "LTS": "formats.time.full",
        "LT": "formats.time.short",
        "L": "formats.date.short",
        "LL": "formats.date.long",
        "LLL": "formats.datetime.long",
        "LLLL": "formats.datetime.full",
    }

    _DEFAULT_DATE_FORMATS: ClassVar[dict[str, str]] = {
        "LTS": "h:mm:ss A",
        "LT": "h:mm A",
        "L": "MM/DD/YYYY",
        "LL": "MMMM D, YYYY",
        "LLL": "MMMM D, YYYY h:mm A",
        "LLLL": "dddd, MMMM D, YYYY h:mm A",
    }

    _REGEX_TOKENS: ClassVar[dict[str, str | Sequence[str] | None]] = {
        "Y": _MATCH_SIGNED,
        "YY": (_MATCH_1_TO_2, _MATCH_2),
        "YYYY": (_MATCH_1_TO_4, _MATCH_4),
        "Q": _MATCH_1,
        "Qo": None,
        "M": _MATCH_1_TO_2,
        "MM": (_MATCH_1_TO_2, _MATCH_2),
        "MMM": _MATCH_WORD,
        "MMMM": _MATCH_WORD,
        "D": _MATCH_1_TO_2,
        "DD": (_MATCH_1_TO_2_LEFT_PAD, _MATCH_2),
        "DDD": _MATCH_1_TO_3,
        "DDDD": _MATCH_3,
        "dddd": _MATCH_WORD,
        "ddd": _MATCH_WORD,
        "dd": _MATCH_WORD,
        "d": _MATCH_1,
        "e": _MATCH_1,
        "E": _MATCH_1,
        "Do": None,
        "H": _MATCH_1_TO_2,
        "HH": (_MATCH_1_TO_2, _MATCH_2),
        "h": _MATCH_1_TO_2,
        "hh": (_MATCH_1_TO_2, _MATCH_2),
        "m": _MATCH_1_TO_2,
        "mm": (_MATCH_1_TO_2, _MATCH_2),
        "s": _MATCH_1_TO_2,
        "ss": (_MATCH_1_TO_2, _MATCH_2),
        "S": (_MATCH_1_TO_3, _MATCH_1),
        "SS": (_MATCH_1_TO_3, _MATCH_2),
        "SSS": (_MATCH_1_TO_3, _MATCH_3),
        "SSSS": _MATCH_UNSIGNED,
        "SSSSS": _MATCH_UNSIGNED,
        "SSSSSS": _MATCH_UNSIGNED,
        "x": _MATCH_SIGNED,
        "X": _MATCH_TIMESTAMP,
        "ZZ": _MATCH_SHORT_OFFSET,
        "Z": _MATCH_OFFSET,
        "z": _MATCH_TIMEZONE,
    }

    _PARSE_TOKENS: ClassVar[dict[str, Callable[[str], Any]]] = {
        "YYYY": lambda year: int(year),
        "YY": lambda year: int(year),
        "Q": lambda quarter: int(quarter),
        "MMMM": lambda month: month,
        "MMM": lambda month: month,
        "MM": lambda month: int(month),
        "M": lambda month: int(month),
        "DDDD": lambda day: int(day),
        "DDD": lambda day: int(day),
        "DD": lambda day: int(day),
        "D": lambda day: int(day),
        "dddd": lambda weekday: weekday,
        "ddd": lambda weekday: weekday,
        "dd": lambda weekday: weekday,
        "d": lambda weekday: int(weekday),
        "E": lambda weekday: int(weekday) - 1,
        "HH": lambda hour: int(hour),
        "H": lambda hour: int(hour),
        "hh": lambda hour: int(hour),
        "h": lambda hour: int(hour),
        "mm": lambda minute: int(minute),
        "m": lambda minute: int(minute),
        "ss": lambda second: int(second),
        "s": lambda second: int(second),
        "S": lambda us: int(us) * 100000,
        "SS": lambda us: int(us) * 10000,
        "SSS": lambda us: int(us) * 1000,
        "SSSS": lambda us: int(us) * 100,
        "SSSSS": lambda us: int(us) * 10,
        "SSSSSS": lambda us: int(us),
        "a": lambda meridiem: meridiem,
        "X": lambda ts: float(ts),
        "x": lambda ts: float(ts) / 1e3,
        "ZZ": str,
        "Z": str,
        "z": str,
    }

    def format(
        self, dt: pendulum.DateTime, fmt: str, locale: str | Locale | None = None
    ) -> str:
        loaded_locale: Locale = Locale.load(locale or pendulum.get_locale())

        def replace(match: Match[str]) -> str:
            literal = match.group(1)
            escaped = match.group(2)
            if literal is not None:
                return literal
            if escaped is not None:
                return escaped
            return self._format_token(dt, cast(str, match.group(3)), loaded_locale)

        return self._FORMAT_RE.sub(replace, fmt)

    def _format_token(self, dt: pendulum.DateTime, token: str, locale: Locale) -> str:
        if token in self._DATE_FORMATS:
            fmt = locale.get(f"custom.date_formats.{token}")
            if fmt is None:
                fmt = self._DEFAULT_DATE_FORMATS[token]
            return self.format(dt, fmt, locale)

        if token in self._LOCALIZABLE_TOKENS:
            return self._format_localizable_token(dt, token, locale)

        if token in self._TOKENS_RULES:
            return self._TOKENS_RULES[token](dt)

        if token.startswith("S") and len(token) <= 9:
            return (f"{dt.microsecond:06d}" + "000")[: len(token)]

        if token in ("Z", "ZZ"):
            return self._format_offset(dt, token == "Z")

        if token in ("W", "WW", "w", "ww"):
            week = dt.isocalendar()[1]
            return f"{week:02d}" if len(token) == 2 else str(week)

        if token in ("GG", "GGG", "GGGG", "gg", "ggg", "gggg"):
            year = dt.isocalendar()[0] if token[0] == "G" else dt.year
            if len(token) == 2:
                return f"{year:04d}"[2:]
            if len(token) == 3:
                return f"{year:03d}"
            return f"{year:04d}"

        return token

    def _format_localizable_token(
        self, dt: pendulum.DateTime, token: str, locale: Locale
    ) -> str:
        if token == "Qo":
            return locale.ordinal(dt.quarter)
        if token == "Mo":
            return locale.ordinal(dt.month)
        if token == "DDDo":
            return locale.ordinal(dt.day_of_year)
        if token == "Do":
            return locale.ordinal(dt.day)
        if token == "Wo":
            return locale.ordinal(dt.isocalendar()[1])
        if token == "wo":
            return locale.ordinal(dt.isocalendar()[1])

        if token in ("dddd", "ddd", "dd"):
            width = {
                "dddd": "wide",
                "ddd": "abbreviated",
                "dd": "short",
            }[token]
            return locale.translation(f"days.{width}.{dt.day_of_week}")

        if token == "do":
            return locale.ordinal((dt.day_of_week + 1) % 7)

        if token in ("e", "eo"):
            first_day = locale.get("week_data.first_day", 0)
            try:
                first_day = int(first_day)
            except (TypeError, ValueError):
                first_day = 0
            day = (dt.day_of_week - first_day) % 7
            return locale.ordinal(day) if token == "eo" else str(day)

        if token in ("MMMM", "MMM"):
            width = "wide" if token == "MMMM" else "abbreviated"
            return locale.translation(f"months.{width}.{dt.month}")

        if token in ("A", "a"):
            key = "day_periods.am" if dt.hour < 12 else "day_periods.pm"
            value = locale.translation(key)
            return value.lower() if token == "a" else value

        return token

    def _format_offset(self, dt: pendulum.DateTime, colon: bool) -> str:
        offset = dt.utcoffset()
        if offset is None:
            offset = datetime.timedelta()

        seconds = int(offset.total_seconds())
        sign = "+" if seconds >= 0 else "-"
        seconds = abs(seconds)
        hours, seconds = divmod(seconds, 3600)
        minutes = seconds // 60

        if colon:
            return f"{sign}{hours:02d}:{minutes:02d}"

        return f"{sign}{hours:02d}{minutes:02d}"