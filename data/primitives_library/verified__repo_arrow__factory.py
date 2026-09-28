import calendar
from datetime import date, datetime, timezone
from datetime import tzinfo as dt_tzinfo
from decimal import Decimal
from time import struct_time
from typing import Any, List, Optional, Tuple, Type, Union, overload

from dateutil import tz as dateutil_tz

from arrow import parser
from arrow.arrow import TZ_EXPR, Arrow
from arrow.constants import DEFAULT_LOCALE
from arrow.util import is_timestamp, iso_to_gregorian


class ArrowFactory:
    type: Type[Arrow]

    def __init__(self, type: Type[Arrow] = Arrow) -> None:
        self.type = type

    def now(self, tzinfo: Optional[TZ_EXPR] = None) -> Arrow:
        if tzinfo is None:
            tzinfo = dateutil_tz.tzlocal()
        elif isinstance(tzinfo, str):
            tzinfo = parser.TzinfoParser.parse(tzinfo)

        return self.type.now(tzinfo)

    def utcnow(self) -> Arrow:
        return self.type.utcnow()

    @overload
    def get(
        self,
        *,
        locale: str = DEFAULT_LOCALE,
        tzinfo: Optional[TZ_EXPR] = None,
        normalize_whitespace: bool = False,
    ) -> Arrow:
        ...

    @overload
    def get(
        self,
        __obj: Union[
            Arrow,
            datetime,
            date,
            struct_time,
            dt_tzinfo,
            int,
            float,
            str,
            Tuple[int, int, int],
        ],
        *,
        locale: str = DEFAULT_LOCALE,
        tzinfo: Optional[TZ_EXPR] = None,
        normalize_whitespace: bool = False,
    ) -> Arrow:
        ...

    @overload
    def get(
        self,
        __arg1: Union[datetime, date],
        __arg2: TZ_EXPR,
        *,
        locale: str = DEFAULT_LOCALE,
        tzinfo: Optional[TZ_EXPR] = None,
        normalize_whitespace: bool = False,
    ) -> Arrow:
        ...

    @overload
    def get(
        self,
        __arg1: str,
        __arg2: Union[str, List[str]],
        *,
        locale: str = DEFAULT_LOCALE,
        tzinfo: Optional[TZ_EXPR] = None,
        normalize_whitespace: bool = False,
    ) -> Arrow:
        ...

    def get(self, *args: Any, **kwargs: Any) -> Arrow:
        arg_count = len(args)
        locale = kwargs.pop("locale", DEFAULT_LOCALE)
        tz = kwargs.get("tzinfo", None)
        normalize_whitespace = kwargs.pop("normalize_whitespace", False)

        if len(kwargs) > 1:
            arg_count = 3

        if len(kwargs) == 1 and tz is None:
            arg_count = 3

        if arg_count == 0:
            if isinstance(tz, str):
                tz = parser.TzinfoParser.parse(tz)
                return self.type.now(tzinfo=tz)

            if isinstance(tz, dt_tzinfo):
                return self.type.now(tzinfo=tz)

            return self.type.utcnow()

        if arg_count == 1:
            arg = args[0]

            if isinstance(arg, Decimal):
                arg = float(arg)

            if arg is None:
                raise TypeError("Cannot parse argument of type None.")

            if not isinstance(arg, str) and is_timestamp(arg):
                if tz is None:
                    tz = timezone.utc
                return self.type.fromtimestamp(arg, tzinfo=tz)

            if isinstance(arg, Arrow):
                return self.type.fromdatetime(arg.datetime, tzinfo=tz)

            if isinstance(arg, datetime):
                return self.type.fromdatetime(arg, tzinfo=tz)

            if isinstance(arg, date):
                return self.type.fromdate(arg, tzinfo=tz)

            if isinstance(arg, dt_tzinfo):
                return self.type.now(tzinfo=arg)

            if isinstance(arg, str):
                dt = parser.DateTimeParser(locale).parse_iso(
                    arg, normalize_whitespace
                )
                return self.type.fromdatetime(dt, tzinfo=tz)

            if isinstance(arg, struct_time):
                return self.type.utcfromtimestamp(calendar.timegm(arg))

            if isinstance(arg, tuple) and len(arg) == 3:
                d = iso_to_gregorian(*arg)
                return self.type.fromdate(d, tzinfo=tz)

            raise TypeError(f"Cannot parse single argument of type {type(arg)!r}.")

        if arg_count == 2:
            arg_1, arg_2 = args[0], args[1]

            if isinstance(arg_1, datetime):
                if isinstance(arg_2, (dt_tzinfo, str)):
                    return self.type.fromdatetime(arg_1, tzinfo=arg_2)
                raise TypeError(
                    f"Cannot parse second argument of type {type(arg_2)!r}."
                )

            if isinstance(arg_1, date):
                if isinstance(arg_2, (dt_tzinfo, str)):
                    return self.type.fromdate(arg_1, tzinfo=arg_2)
                raise TypeError(
                    f"Cannot parse second argument of type {type(arg_2)!r}."
                )

            if isinstance(arg_1, str) and isinstance(arg_2, (str, list)):
                dt = parser.DateTimeParser(locale).parse(
                    arg_1, arg_2, normalize_whitespace
                )
                return self.type.fromdatetime(dt, tzinfo=tz)

            raise TypeError(
                f"Cannot parse two arguments of types {type(arg_1)!r} and "
                f"{type(arg_2)!r}."
            )

        return self.type(*args, **kwargs)