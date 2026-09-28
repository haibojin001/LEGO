# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg86::calendar.isleap+datetime.date+datetime.timedelta
# name: calendar_datetime_primitive
# summary: Uses calendar.isleap, datetime.date, datetime.timedelta across 2 repos
# anchor_symbols: ['calendar.isleap', 'datetime.date', 'datetime.timedelta']
# observed in 2 repos: ['akfamily__akshare', 'turicas__rows']...

# --- from turicas__rows::rows/utils/date.py::next_year ---
def next_year(date):
    """
    >>> next_year(datetime.date(2020, 2, 28))
    datetime.date(2021, 2, 28)
    >>> next_year(datetime.date(2016, 2, 29))
    datetime.date(2017, 3, 1)
    """
    # TODO: add `semantic` as in `next_month`

    days = 365 if not isleap(date.year) else 366
    return date + datetime.timedelta(days=days)

# --- from turicas__rows::rows/utils/date.py::last_year ---
def last_year(date):
    """
    >>> next_year(datetime.date(2020, 2, 28))
    datetime.date(2021, 2, 28)
    >>> next_year(datetime.date(2016, 2, 29))
    datetime.date(2017, 3, 1)
    """
    # TODO: add `semantic` as in `next_month`

    days = 365 if not isleap(date.year - 1) else 366
    return date - datetime.timedelta(days=days)

# --- from akfamily__akshare::akshare/futures/cons.py::get_latest_data_date ---
def get_latest_data_date(day):
    """
    获取最新的有数据的交易日
    :param day: datetime.datetime
    :return string YYYYMMDD
    """
    calendar = get_calendar()
    if day.strftime("%Y%m%d") in calendar:
        if day.time() > datetime.time(17, 0, 0):
            return day.strftime("%Y%m%d")
        else:
            return last_trading_day(day.strftime("%Y%m%d"))
    else:
        while day.strftime("%Y%m%d") not in calendar:
            day = day - datetime.timedelta(days=1)
        return day.strftime("%Y%m%d")

# --- from akfamily__akshare::akshare/option/cons.py::get_latest_data_date ---
def get_latest_data_date(day):
    """
    获取最新的有数据的交易日
    :param day: datetime.datetime
    :return string YYYYMMDD
    """
    calendar = get_calendar()
    if day.strftime("%Y%m%d") in calendar:
        if day.time() > datetime.time(17, 0, 0):
            return day.strftime("%Y%m%d")
        else:
            return last_trading_day(day.strftime("%Y%m%d"))
    else:
        while day.strftime("%Y%m%d") not in calendar:
            day = day - datetime.timedelta(days=1)
        return day.strftime("%Y%m%d")
