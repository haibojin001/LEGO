from __future__ import annotations

import typing

T = typing.TypeVar("T")

StrSequenceOrSet: typing.TypeAlias = typing.Sequence[str] | typing.AbstractSet[str]
Validator: typing.TypeAlias = typing.Callable[[typing.Any], typing.Any]
ErrorMessageValue: typing.TypeAlias = str | list | dict
ErrorMessages: typing.TypeAlias = dict[str, ErrorMessageValue]
UnknownOption: typing.TypeAlias = typing.Literal["exclude", "include", "raise"]

PreLoadCallable = typing.Callable[[typing.Any], typing.Any]
PostLoadCallable = typing.Callable[[T], T]


class SchemaValidator(typing.Protocol):
    def __call__(
        self,
        output: typing.Any,
        original_data: typing.Any = ...,
        *,
        partial: bool | StrSequenceOrSet | None = None,
        unknown: UnknownOption | None = None,
        many: bool = False,
    ) -> None: ...


class RenderModule(typing.Protocol):
    def dumps(
        self,
        obj: typing.Any,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> str: ...

    def loads(
        self,
        s: str | bytes | bytearray,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any: ...