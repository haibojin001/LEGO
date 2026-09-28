from datekit.duration import _coerce_date, _dt

__all__ = ["is_weekday", "business_days_between", "add_business_days"]


def is_weekday(d) -> bool:
    d = _coerce_date(d)
    return d.weekday() < 5


def business_days_between(a, b) -> int:
    a = _coerce_date(a)
    b = _coerce_date(b)

    span = (b - a).days
    if span <= 0:
        return 0

    weeks, extra_days = divmod(span, 7)
    count = weeks * 5

    start_weekday = a.weekday()
    for i in range(extra_days):
        if (start_weekday + i) % 7 < 5:
            count += 1

    return count


def add_business_days(d, n: int):
    d = _coerce_date(d)

    try:
        n = n.__index__()
    except AttributeError:
        raise TypeError("'n' must be an integer") from None

    if n < 0:
        raise ValueError("'n' must be >= 0")

    if n == 0:
        return d

    weekday = d.weekday()

    if weekday >= 5:
        days_to_monday = 7 - weekday
        d = d + _dt.timedelta(days=days_to_monday)
        n -= 1
        if n == 0:
            return d

    weeks, extra_business_days = divmod(n, 5)
    d = d + _dt.timedelta(days=weeks * 7)

    if extra_business_days:
        weekday = d.weekday()
        calendar_days = extra_business_days
        if weekday + extra_business_days >= 5:
            calendar_days += 2
        d = d + _dt.timedelta(days=calendar_days)

    return d