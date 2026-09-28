from __future__ import annotations

import contextlib as _contextlib
import contextvars as _contextvars
import typing as _typing

try:
    from types import EllipsisType
except ImportError:
    EllipsisType = type(...)

_ContextT = _typing.TypeVar("_ContextT")
_DefaultT = _typing.TypeVar("_DefaultT")

_CURRENT_CONTEXT: _contextvars.ContextVar = _contextvars.ContextVar("context")


class Context(_contextlib.AbstractContextManager, _typing.Generic[_ContextT]):
    """Context manager used to make context available within a scope."""

    def __init__(self, context: _ContextT) -> None:
        self.context = context
        self.token: _contextvars.Token | None = None

    def __enter__(self) -> Context[_ContextT]:
        self.token = _CURRENT_CONTEXT.set(self.context)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        _CURRENT_CONTEXT.reset(_typing.cast(_contextvars.Token, self.token))

    @classmethod
    def get(cls, default: _DefaultT | EllipsisType = ...) -> _ContextT | _DefaultT:
        """Return the active context, or a supplied fallback value."""
        if default is ...:
            return _CURRENT_CONTEXT.get()
        return _CURRENT_CONTEXT.get(default)