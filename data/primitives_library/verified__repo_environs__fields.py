from __future__ import annotations

import logging
import pathlib
import re
import typing
from datetime import timedelta
from urllib.parse import ParseResult, urlparse

from marshmallow import ValidationError, fields


class Path(fields.Field[pathlib.Path]):
    def _serialize(self, value: pathlib.Path | None, *args, **kwargs) -> str | None:
        if value is None:
            return None
        return str(value)

    def _deserialize(self, value, *args, **kwargs) -> pathlib.Path:
        if isinstance(value, pathlib.Path):
            return value
        value = super()._deserialize(value, *args, **kwargs)
        return pathlib.Path(value)


class LogLevel(fields.Integer):
    def _format_num(self, value) -> int:
        try:
            return super()._format_num(value)
        except (TypeError, ValueError) as exc:
            name = value.upper()
            if hasattr(logging, name) and isinstance(getattr(logging, name), int):
                return getattr(logging, name)
            raise ValidationError("Not a valid log level.") from exc


class TimeDelta(fields.TimeDelta):
    """A timedelta field supporting ordered and ISO 8601 duration strings."""

    DEFAULT_FORMAT = "gep2257"

    _GEP_2257_REGEX = re.compile(
        r"^(?:\s*)"
        r"(?:(-?\d+)\s*w\s*)?"
        r"(?:(-?\d+)\s*d\s*)?"
        r"(?:(-?\d+)\s*h\s*)?"
        r"(?:(-?\d+)\s*m\s*)?"
        r"(?:(-?\d+)\s*s\s*)?"
        r"(?:(-?\d+)\s*ms\s*)?"
        r"(?:(-?\d+)\s*[µu]s\s*)?$"
    )

    _ISO_8601_REGEX = re.compile(
        r"^(?:\s*)"
        r"(?P<sign>[+-]?)"
        r"P"
        r"(?:(?P<weeks>\d+(?:\.\d+)?)W|"
        r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
        r")?)\s*$"
    )

    def __init__(
        self,
        format: typing.Literal["gep2257", "iso8601"] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.format = format

    def _deserialize(self, value, *args, **kwargs) -> timedelta:
        if isinstance(value, timedelta):
            return value

        if isinstance(value, str):
            data_format = self.format or self.DEFAULT_FORMAT

            if data_format == "gep2257":
                match = self._GEP_2257_REGEX.match(value)
                if match is not None and any(groups := match.groups(default=0)):
                    return timedelta(
                        weeks=int(groups[0]),
                        days=int(groups[1]),
                        hours=int(groups[2]),
                        minutes=int(groups[3]),
                        seconds=int(groups[4]),
                        milliseconds=int(groups[5]),
                        microseconds=int(groups[6]),
                    )

            elif data_format == "iso8601":
                match = self._ISO_8601_REGEX.match(value)
                if match is not None:
                    parts = match.groupdict()
                    sign = -1.0 if parts.get("sign") == "-" else 1.0
                    seconds = 0.0

                    if parts.get("weeks") is not None:
                        seconds += float(parts["weeks"]) * 7 * 24 * 3600
                    else:
                        if parts.get("days") is not None:
                            seconds += float(parts["days"]) * 24 * 3600
                        if parts.get("hours") is not None:
                            seconds += float(parts["hours"]) * 3600
                        if parts.get("minutes") is not None:
                            seconds += float(parts["minutes"]) * 60
                        if parts.get("seconds") is not None:
                            seconds += float(parts["seconds"])

                    return timedelta(seconds=sign * seconds)

        return super()._deserialize(value, *args, **kwargs)


class Url(fields.Url):
    """A URL field that deserializes values into urllib parse results."""

    def _serialize(self, value: ParseResult, *args, **kwargs) -> str:
        return value.geturl()

    def deserialize(
        self,
        value: typing.Any,
        attr: str | None = None,
        data: typing.Mapping[str, typing.Any] | None = None,
        **kwargs,
    ) -> ParseResult:
        if isinstance(value, ParseResult):
            return value
        result = typing.cast(str, super().deserialize(value, attr, data, **kwargs))
        return urlparse(result)