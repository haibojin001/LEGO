from __future__ import annotations

import contextlib
import contextvars
import typing

try:
    from types import EllipsisType
except ImportError:
    EllipsisType = type(Ellipsis)  # type: ignore[misc]

_ContextT = typing.TypeVar("_ContextT")
_DefaultT = typing.TypeVar("_DefaultT")

_CURRENT_CONTEXT: contextvars.ContextVar = contextvars.ContextVar("context")


class Context(contextlib.AbstractContextManager, typing.Generic[_ContextT]):
    """Context manager used to make context available within a scope."""

    def __init__(self, context: _ContextT) -> None:
        self.context = context
        self.token: contextvars.Token | None = None

    def __enter__(self) -> Context[_ContextT]:
        self.token = _CURRENT_CONTEXT.set(self.context)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        _CURRENT_CONTEXT.reset(typing.cast(contextvars.Token, self.token))

    @classmethod
    def get(cls, default: _DefaultT | EllipsisType = ...) -> _ContextT | _DefaultT:
        """Return the active context, or a supplied fallback value."""
        if default is ...:
            return _CURRENT_CONTEXT.get()
        return _CURRENT_CONTEXT.get(default)