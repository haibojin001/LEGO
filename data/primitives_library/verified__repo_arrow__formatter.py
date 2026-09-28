import re
from datetime import datetime, timedelta, timezone
from typing import Final, Optional, Pattern, cast

from arrow import locales
from arrow.constants import DEFAULT_LOCALE

FORMAT_ATOM: Final[str] = "YYYY-MM-DD HH:mm:ssZZ"
FORMAT_COOKIE: Final[str] = "dddd, DD-MMM-YYYY HH:mm:ss ZZZ"
FORMAT_RFC822: Final[str] = "ddd, DD MMM YY HH:mm:ss Z"
FORMAT_RFC850: Final[str] = "dddd, DD-MMM-YY HH:mm:ss ZZZ"
FORMAT_RFC1036: Final[str] = "ddd, DD MMM YY HH:mm:ss Z"
FORMAT_RFC1123: Final[str] = "ddd, DD MMM YYYY HH:mm:ss Z"
FORMAT_RFC2822: Final[str] = "ddd, DD MMM YYYY HH:mm:ss Z"
FORMAT_RFC3339: Final[str] = "YYYY-MM-DD HH:mm:ssZZ"
FORMAT_RFC3339_STRICT: Final[str] = "YYYY-MM-DDTHH:mm:ssZZ"
FORMAT_RSS: Final[str] = "ddd, DD MMM YYYY HH:mm:ss Z"
FORMAT_W3C: Final[str] = "YYYY-MM-DD HH:mm:ssZZ"


class DateTimeFormatter:
    _FORMAT_RE: Final[Pattern[str]] = re.compile(
        r"(\[(?:(?=(?P<literal>[^]]))(?P=literal))*\]|"
        r"YYY?Y?|MM?M?M?|Do|DD?D?D?|d?dd?d?|HH?|hh?|mm?|ss?|"
        r"SS?S?S?S?S?|ZZ?Z?|a|A|X|x|W)"
    )

    locale: locales.Locale

    def __init__(self, locale: str = DEFAULT_LOCALE) -> None:
        self.locale = locales.get_locale(locale)

    def format(self, dt: datetime, fmt: str) -> str:
        def render(match: re.Match[str]) -> str:
            value = self._format_token(dt, match.group(0))
            return "" if value is None else value

        return self._FORMAT_RE.sub(render, fmt)

    def _format_token(self, dt: datetime, token: Optional[str]) -> Optional[str]:
        if token is None:
            return None

        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]

        if token == "YYYY":
            return self.locale.year_full(dt.year)
        if token == "YY":
            return self.locale.year_abbreviation(dt.year)

        if token == "MMMM":
            return self.locale.month_name(dt.month)
        if token == "MMM":
            return self.locale.month_abbreviation(dt.month)
        if token == "MM":
            return f"{dt.month:02d}"
        if token == "M":
            return str(dt.month)

        if token == "DDDD":
            return f"{dt.timetuple().tm_yday:03d}"
        if token == "DDD":
            return str(dt.timetuple().tm_yday)
        if token == "DD":
            return f"{dt.day:02d}"
        if token == "D":
            return str(dt.day)
        if token == "Do":
            return self.locale.ordinal_number(dt.day)

        if token == "dddd":
            return self.locale.day_name(dt.isoweekday())
        if token == "ddd":
            return self.locale.day_abbreviation(dt.isoweekday())
        if token == "d":
            return str(dt.isoweekday())

        if token == "HH":
            return f"{dt.hour:02d}"
        if token == "H":
            return str(dt.hour)

        if token in ("hh", "h"):
            hour = dt.hour if 0 < dt.hour < 13 else abs(dt.hour - 12)
            return f"{hour:02d}" if token == "hh" else str(hour)

        if token == "mm":
            return f"{dt.minute:02d}"
        if token == "m":
            return str(dt.minute)

        if token == "ss":
            return f"{dt.second:02d}"
        if token == "s":
            return str(dt.second)

        fractions = {
            "SSSSSS": (1, 6),
            "SSSSS": (10, 5),
            "SSSS": (100, 4),
            "SSS": (1000, 3),
            "SS": (10000, 2),
            "S": (100000, 1),
        }
        if token in fractions:
            divisor, width = fractions[token]
            return f"{dt.microsecond // divisor:0{width}d}"

        if token == "X":
            return str(dt.timestamp())

        if token == "x":
            return f"{dt.timestamp() * 1_000_000:.0f}"

        if token == "ZZZ":
            return dt.tzname()

        if token == "Z" or token == "ZZ":
            colon = ":" if token == "ZZ" else ""
            info = timezone.utc if dt.tzinfo is None else dt.tzinfo
            offset = cast(timedelta, info.utcoffset(dt))
            minutes = int(offset.total_seconds() / 60)
            sign = "+" if minutes >= 0 else "-"
            hours, remainder = divmod(abs(minutes), 60)
            return f"{sign}{hours:02d}{colon}{remainder:02d}"

        if token == "a" or token == "A":
            return self.locale.meridian(dt.hour, token)

        if token == "W":
            year, week, weekday = dt.isocalendar()
            return f"{year}-W{week:02d}-{weekday}"

        return None