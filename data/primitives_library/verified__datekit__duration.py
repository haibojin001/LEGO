from datekit.parse import _dt, parse_date

__all__ = ["days_between", "add_days"]


def _coerce_date(value):
    if isinstance(value, str):
        value = parse_date(value)

    if isinstance(value, _dt.datetime):
        return value.date()

    if isinstance(value, _dt.date):
        return value

    raise TypeError("expected a date object or ISO date string")


def days_between(a, b) -> int:
    return (_coerce_date(b) - _coerce_date(a)).days


def add_days(d, n: int):
    if not isinstance(n, int):
        raise TypeError("n must be an int")

    return _coerce_date(d) + _dt.timedelta(days=n)