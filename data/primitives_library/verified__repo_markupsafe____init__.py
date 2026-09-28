from __future__ import annotations

import collections.abc as cabc
import string
import typing as t

try:
    from ._speedups import _escape_inner
except ImportError:
    try:
        from ._native import _escape_inner
    except ImportError:
        def _escape_inner(s: str) -> str:
            return (
                s.replace("&", "&amp;")
                .replace(">", "&gt;")
                .replace("<", "&lt;")
                .replace("'", "&#39;")
                .replace('"', "&#34;")
            )

if t.TYPE_CHECKING:
    import typing_extensions as te


class _HasHTML(t.Protocol):
    def __html__(self, /) -> str: ...


class _TPEscape(t.Protocol):
    def __call__(self, s: t.Any, /) -> Markup: ...


def escape(s: t.Any, /) -> Markup:
    if type(s) is str:
        return Markup(_escape_inner(s))

    if hasattr(s, "__html__"):
        return Markup(s.__html__())

    return Markup(_escape_inner(str(s)))


def escape_silent(s: t.Any | None, /) -> Markup:
    if s is None:
        return Markup()
    return escape(s)


def soft_str(s: t.Any, /) -> str:
    if not isinstance(s, str):
        return str(s)
    return s


class _MarkupEscapeHelper:
    __slots__ = ("obj", "escape")

    def __init__(self, obj: t.Any, escape: _TPEscape) -> None:
        self.obj = obj
        self.escape = escape

    def __str__(self) -> str:
        return self.escape(self.obj)

    def __repr__(self) -> str:
        return repr(self.escape(self.obj))

    def __format__(self, format_spec: str) -> str:
        if hasattr(self.obj, "__html_format__"):
            return self.obj.__html_format__(format_spec)
        return format(self.escape(self.obj), format_spec)

    def __int__(self) -> int:
        return int(self.obj)

    def __float__(self) -> float:
        return float(self.obj)

    def __getitem__(self, key: t.Any) -> t.Any:
        return _MarkupEscapeHelper(self.obj[key], self.escape)


class _MarkupEscapeMapping:
    __slots__ = ("mapping", "escape")

    def __init__(self, mapping: t.Mapping[t.Any, t.Any], escape: _TPEscape) -> None:
        self.mapping = mapping
        self.escape = escape

    def __getitem__(self, key: t.Any) -> _MarkupEscapeHelper:
        return _MarkupEscapeHelper(self.mapping[key], self.escape)


class _EscapeFormatter(string.Formatter):
    __slots__ = ("escape",)

    def __init__(self, escape: _TPEscape) -> None:
        self.escape = escape

    def format_field(self, value: t.Any, format_spec: str) -> str:
        if hasattr(value, "__html_format__"):
            return value.__html_format__(format_spec)
        return format(self.escape(value), format_spec)


class Markup(str):
    __slots__ = ()

    def __new__(
        cls,
        object: t.Any = "",
        encoding: str | None = None,
        errors: str = "strict",
    ) -> te.Self:
        if hasattr(object, "__html__"):
            object = object.__html__()

        if encoding is None:
            return super().__new__(cls, object)

        return super().__new__(cls, object, encoding, errors)

    def __html__(self, /) -> te.Self:
        return self

    def __add__(self, value: str | _HasHTML, /) -> te.Self:
        if isinstance(value, str) or hasattr(value, "__html__"):
            return self.__class__(super().__add__(self.escape(value)))
        return NotImplemented

    def __radd__(self, value: str | _HasHTML, /) -> te.Self:
        if isinstance(value, str) or hasattr(value, "__html__"):
            return self.escape(value).__add__(self)
        return NotImplemented

    def __mul__(self, value: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().__mul__(value))

    def __rmul__(self, value: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().__mul__(value))

    def __mod__(self, value: t.Any, /) -> te.Self:
        if isinstance(value, tuple):
            value = tuple(_MarkupEscapeHelper(x, self.escape) for x in value)
        elif hasattr(type(value), "__getitem__") and not isinstance(value, str):
            value = _MarkupEscapeHelper(value, self.escape)
        else:
            value = (_MarkupEscapeHelper(value, self.escape),)

        return self.__class__(super().__mod__(value))

    def __repr__(self, /) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    def join(self, iterable: cabc.Iterable[str | _HasHTML], /) -> te.Self:
        return self.__class__(super().join(map(self.escape, iterable)))

    def split(
        self,
        /,
        sep: str | None = None,
        maxsplit: t.SupportsIndex = -1,
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().split(sep, maxsplit)]

    def rsplit(
        self,
        /,
        sep: str | None = None,
        maxsplit: t.SupportsIndex = -1,
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().rsplit(sep, maxsplit)]

    def splitlines(self, /, keepends: bool = False) -> list[te.Self]:
        return [self.__class__(v) for v in super().splitlines(keepends)]

    def unescape(self, /) -> str:
        from html import unescape
        return unescape(str(self))

    def striptags(self, /) -> str:
        value = str(self)

        while (start := value.find("<!--")) != -1:
            if (end := value.find("-->", start)) == -1:
                break
            value = f"{value[:start]}{value[end + 3:]}"

        while (start := value.find("<")) != -1:
            if (end := value.find(">", start)) == -1:
                break
            value = f"{value[:start]}{value[end + 1:]}"

        value = " ".join(value.split())
        return self.__class__(value).unescape()

    @classmethod
    def escape(cls, s: t.Any, /) -> te.Self:
        rv = escape(s)
        if rv.__class__ is not cls:
            return cls(rv)
        return rv

    def __getitem__(self, key: t.SupportsIndex | slice, /) -> te.Self:
        return self.__class__(super().__getitem__(key))

    def capitalize(self, /) -> te.Self:
        return self.__class__(super().capitalize())

    def title(self, /) -> te.Self:
        return self.__class__(super().title())

    def lower(self, /) -> te.Self:
        return self.__class__(super().lower())

    def upper(self, /) -> te.Self:
        return self.__class__(super().upper())

    def casefold(self, /) -> te.Self:
        return self.__class__(super().casefold())

    def swapcase(self, /) -> te.Self:
        return self.__class__(super().swapcase())

    def replace(
        self,
        old: str,
        new: str,
        count: t.SupportsIndex = -1,
        /,
    ) -> te.Self:
        return self.__class__(super().replace(old, self.escape(new), count))

    def ljust(
        self,
        width: t.SupportsIndex,
        fillchar: str = " ",
        /,
    ) -> te.Self:
        return self.__class__(super().ljust(width, self.escape(fillchar)))

    def rjust(
        self,
        width: t.SupportsIndex,
        fillchar: str = " ",
        /,
    ) -> te.Self:
        return self.__class__(super().rjust(width, self.escape(fillchar)))

    def center(
        self,
        width: t.SupportsIndex,
        fillchar: str = " ",
        /,
    ) -> te.Self:
        return self.__class__(super().center(width, self.escape(fillchar)))

    def lstrip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().lstrip(chars))

    def rstrip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().rstrip(chars))

    def strip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().strip(chars))

    def zfill(self, width: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().zfill(width))

    def translate(self, table: cabc.Mapping[int, str | int | None], /) -> te.Self:
        return self.__class__(super().translate(table))

    def removeprefix(self, prefix: str, /) -> te.Self:
        return self.__class__(super().removeprefix(prefix))

    def removesuffix(self, suffix: str, /) -> te.Self:
        return self.__class__(super().removesuffix(suffix))

    def partition(self, sep: str, /) -> tuple[te.Self, te.Self, te.Self]:
        return tuple(self.__class__(v) for v in super().partition(sep))

    def rpartition(self, sep: str, /) -> tuple[te.Self, te.Self, te.Self]:
        return tuple(self.__class__(v) for v in super().rpartition(sep))

    def format(self, *args: t.Any, **kwargs: t.Any) -> te.Self:
        formatter = _EscapeFormatter(self.escape)
        return self.__class__(formatter.vformat(self, args, kwargs))

    def format_map(self, mapping: cabc.Mapping[str, t.Any], /) -> te.Self:
        formatter = _EscapeFormatter(self.escape)
        return self.__class__(formatter.vformat(self, (), mapping))


__all__ = ["Markup", "escape", "escape_silent", "soft_str"]