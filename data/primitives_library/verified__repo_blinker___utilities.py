from __future__ import annotations

import collections.abc as c
import inspect
import typing as t
from weakref import WeakMethod
from weakref import ref

T = t.TypeVar("T")


class Symbol:
    """A named singleton value."""

    symbols: t.ClassVar[dict[str, Symbol]] = {}

    def __new__(cls, name: str) -> Symbol:
        try:
            return cls.symbols[name]
        except KeyError:
            instance = super().__new__(cls)
            cls.symbols[name] = instance
            return instance

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __getnewargs__(self) -> tuple[t.Any, ...]:
        return (self.name,)


def make_id(obj: object) -> c.Hashable:
    if inspect.ismethod(obj):
        return (id(obj.__func__), id(obj.__self__))

    if isinstance(obj, (str, int)):
        return obj

    return id(obj)


def make_ref(
    obj: T, callback: c.Callable[[ref[T]], None] | None = None
) -> ref[T]:
    if inspect.ismethod(obj):
        return WeakMethod(obj, callback)  # type: ignore[arg-type, return-value]

    return ref(obj, callback)