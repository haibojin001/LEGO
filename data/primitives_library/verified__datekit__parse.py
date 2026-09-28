import datetime as _dt

__all__ = ["parse_date", "parse_datetime", "iso"]


def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _parse_ndigits(s: str, start: int, end: int, field: str) -> int:
    if end > len(s):
        raise ValueError("malformed ISO-8601 string: missing " + field)

    value = 0
    for i in range(start, end):
        ch = s[i]
        if not _is_ascii_digit(ch):
            raise ValueError("malformed ISO-8601 string: invalid " + field)
        value = value * 10 + (ord(ch) - 48)
    return value


def parse_date(s: str) -> _dt.date:
    """Parse an ISO-8601 calendar date in exactly ``YYYY-MM-DD`` form."""
    if not isinstance(s, str):
        raise ValueError("malformed ISO-8601 date: expected str")

    if len(s) != 10:
        raise ValueError("malformed ISO-8601 date: expected YYYY-MM-DD")

    if s[4] != "-" or s[7] != "-":
        raise ValueError("malformed ISO-8601 date: expected YYYY-MM-DD")

    year = _parse_ndigits(s, 0, 4, "year")
    month = _parse_ndigits(s, 5, 7, "month")
    day = _parse_ndigits(s, 8, 10, "day")

    try:
        return _dt.date(year, month, day)
    except ValueError as exc:
        raise ValueError("malformed ISO-8601 date: " + str(exc)) from None


def parse_datetime(s: str) -> _dt.datetime:
    """
    Parse an ISO-8601 datetime in exactly ``YYYY-MM-DDTHH:MM:SS`` form.

    A single space may be used instead of ``T`` as the date/time separator.
    """
    if not isinstance(s, str):
        raise ValueError("malformed ISO-8601 datetime: expected str")

    if len(s) != 19:
        raise ValueError(
            "malformed ISO-8601 datetime: expected YYYY-MM-DDTHH:MM:SS"
        )

    if s[10] not in ("T", " "):
        raise ValueError(
            "malformed ISO-8601 datetime: expected T or space separator"
        )

    if s[4] != "-" or s[7] != "-" or s[13] != ":" or s[16] != ":":
        raise ValueError(
            "malformed ISO-8601 datetime: expected YYYY-MM-DDTHH:MM:SS"
        )

    year = _parse_ndigits(s, 0, 4, "year")
    month = _parse_ndigits(s, 5, 7, "month")
    day = _parse_ndigits(s, 8, 10, "day")
    hour = _parse_ndigits(s, 11, 13, "hour")
    minute = _parse_ndigits(s, 14, 16, "minute")
    second = _parse_ndigits(s, 17, 19, "second")

    try:
        return _dt.datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ValueError("malformed ISO-8601 datetime: " + str(exc)) from None


def iso(d) -> str:
    """Return ``d.isoformat()``."""
    return d.isoformat()