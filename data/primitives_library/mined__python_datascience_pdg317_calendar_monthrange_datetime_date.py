# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg317::calendar.monthrange+datetime.date
# name: calendar_datetime_primitive
# summary: Uses calendar.monthrange, datetime.date across 2 repos
# anchor_symbols: ['calendar.monthrange', 'datetime.date']
# observed in 2 repos: ['jackzhenguo__python-small-examples', 'meteostat__meteostat']...

# --- from meteostat__meteostat::meteostat/utils/parsers.py::parse_month ---
def parse_month(
    value: datetime.date | datetime.datetime | None, is_end: bool = False
) -> datetime.date | None:
    """
    Convert a given date/time input to the first or last day of the month respectively
    """
    if not value:
        return None

    last_day = calendar.monthrange(value.year, value.month)[1]

    return datetime.date(value.year, value.month, last_day if is_end else 1)

# --- from jackzhenguo__python-small-examples::md/batch.py::getEverydaySince ---
def getEverydaySince(year,month,day,n=10):
    i = 0
    _, days = calendar.monthrange(year, month)
    while i < n: 
        d = date(year,month,day)    
        if day == days:
            month,day = month+1,0
            _, days = calendar.monthrange(year, month)
            if month == 13:
                year,month = year+1,1
                _, days = calendar.monthrange(year, month)
        yield d
        day += 1
        i += 1
