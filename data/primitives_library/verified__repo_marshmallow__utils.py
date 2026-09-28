from __future__ import annotations

import datetime as dt
import inspect
import typing
from collections.abc import Mapping, Sequence

from marshmallow.constants import missing


def is_generator(obj) -> typing.TypeGuard[typing.Generator]:
    return inspect.isgenerator(obj) or inspect.isgeneratorfunction(obj)


def is_iterable_but_not_string(obj) -> typing.TypeGuard[typing.Iterable]:
    return is_generator(obj) or (
        hasattr(obj, "__iter__") and not hasattr(obj, "strip")
    )


def is_sequence_but_not_string(obj) -> typing.TypeGuard[Sequence]:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def is_collection(obj) -> typing.TypeGuard[typing.Iterable]:
    return is_iterable_but_not_string(obj) and not isinstance(obj, Mapping)


def is_aware(datetime: dt.datetime) -> bool:
    zone = datetime.tzinfo
    return zone is not None and zone.utcoffset(datetime) is not None


def from_timestamp(value: typing.Any) -> dt.datetime:
    if value is True or value is False:
        raise ValueError("Not a valid POSIX timestamp")

    seconds = float(value)
    if seconds < 0:
        raise ValueError("Not a valid POSIX timestamp")

    try:
        converted = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except OverflowError as error:
        raise ValueError("Timestamp is too large") from error
    except OSError as error:
        raise ValueError("Error converting value to datetime") from error

    return converted.replace(tzinfo=None)


def from_timestamp_ms(value: typing.Any) -> dt.datetime:
    if value is True or value is False:
        raise ValueError("Not a valid POSIX timestamp")
    return from_timestamp(float(value) / 1000)


def timestamp(value: dt.datetime) -> float:
    if not is_aware(value):
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def timestamp_ms(value: dt.datetime) -> float:
    return timestamp(value) * 1000


def ensure_text_type(val: str | bytes) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def pluck(dictlist: list[dict[str, typing.Any]], key: str):
    return [item[key] for item in dictlist]


def get_value(obj, key: str, default=missing):
    if "." not in key:
        return _get_value_for_key(obj, key, default)

    current = obj
    for part in key.split("."):
        current = _get_value_for_key(current, part, default)
    return current


def _get_value_for_key(obj, key, default):
    if not hasattr(obj, "__getitem__"):
        return getattr(obj, key, default)

    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return getattr(obj, key, default)


def set_value(dct: dict[str, typing.Any], key: str, value: typing.Any):
    if "." not in key:
        dct[key] = value
        return

    first, remaining = key.split(".", 1)
    child = dct.setdefault(first, {})
    if not isinstance(child, dict):
        raise ValueError(
            f"Cannot set {key} in {first} due to existing value: {child}"
        )
    set_value(child, remaining, value)


def callable_or_raise(obj):
    if not callable(obj):
        raise TypeError(f"Object {obj!r} is not callable.")
    return obj


def timedelta_to_microseconds(value: dt.timedelta) -> int:
    return (value.days * (24 * 3600) + value.seconds) * 1000000 + value.microseconds