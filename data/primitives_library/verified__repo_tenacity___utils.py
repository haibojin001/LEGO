import contextlib as _contextlib
import functools as _functools
import inspect as _inspect
import sys as _sys
import typing as _typing
from datetime import timedelta

if _typing.TYPE_CHECKING:
    from typing_extensions import override as override
elif _sys.version_info >= (3, 12):
    from typing import override
else:
    _F = _typing.TypeVar("_F", bound=_typing.Callable[..., _typing.Any])

    def override(method: _F) -> _F:
        """Backport of `typing.override` for Python < 3.12."""
        with _contextlib.suppress(AttributeError, TypeError):
            setattr(method, "__override__", True)
        return method


MAX_WAIT = _sys.maxsize / 2


class LoggerProtocol(_typing.Protocol):
    """
    Protocol used by utilities that consume logger-like objects.
    """

    def log(self, level: int, msg: str, *args: _typing.Any) -> _typing.Any:
        ...


def find_ordinal(pos_num: int) -> str:
    remainder = pos_num % 100
    if 11 <= remainder <= 13:
        return "th"
    if pos_num == 0:
        return "th"
    if pos_num == 1:
        return "st"
    if pos_num == 2:
        return "nd"
    if pos_num == 3:
        return "rd"
    if 4 <= pos_num <= 20:
        return "th"
    return find_ordinal(pos_num % 10)


def to_ordinal(pos_num: int) -> str:
    suffix = find_ordinal(pos_num)
    return f"{pos_num}{suffix}"


def get_callback_name(cb: _typing.Callable[..., _typing.Any]) -> str:
    """Return the most descriptive qualified name available for a callback."""
    names = []
    try:
        names.append(cb.__qualname__)
    except AttributeError:
        with _contextlib.suppress(AttributeError):
            names.append(cb.__name__)

    if not names:
        return repr(cb)

    with _contextlib.suppress(AttributeError):
        module_name = cb.__module__
        if module_name:
            names.insert(0, module_name)

    return ".".join(names)


time_unit_type = int | float | timedelta


def to_seconds(time_unit: time_unit_type) -> float:
    if isinstance(time_unit, timedelta):
        time_unit = time_unit.total_seconds()
    return float(time_unit)


def is_coroutine_callable(call: _typing.Callable[..., _typing.Any]) -> bool:
    if _inspect.isclass(call):
        return False
    if _inspect.iscoroutinefunction(call):
        return True

    if isinstance(call, _functools.partial):
        candidate = call.func
    else:
        candidate = getattr(call, "__call__", None)
    return _inspect.iscoroutinefunction(candidate)


def wrap_to_async_func(
    call: _typing.Callable[..., _typing.Any],
) -> _typing.Callable[..., _typing.Awaitable[_typing.Any]]:
    if is_coroutine_callable(call):
        return call

    async def inner(*args: _typing.Any, **kwargs: _typing.Any) -> _typing.Any:
        return call(*args, **kwargs)

    return inner