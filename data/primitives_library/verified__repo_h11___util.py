from __future__ import annotations

from typing import Any, Dict, NoReturn, Pattern, Tuple, Type, TypeVar, Union

__all__ = [
    "ProtocolError",
    "LocalProtocolError",
    "RemoteProtocolError",
    "validate",
    "bytesify",
]


class ProtocolError(Exception):
    def __init__(self, msg: str, error_status_hint: int = 400) -> None:
        if self.__class__ is ProtocolError:
            raise TypeError("tried to directly instantiate ProtocolError")
        super().__init__(msg)
        self.error_status_hint = error_status_hint


class LocalProtocolError(ProtocolError):
    def _reraise_as_remote_protocol_error(self) -> NoReturn:
        self.__class__ = RemoteProtocolError  # type: ignore[misc]
        raise self


class RemoteProtocolError(ProtocolError):
    pass


def validate(
    regex: Pattern[bytes], data: bytes, msg: str = "malformed data", *format_args: Any
) -> Dict[str, bytes]:
    result = regex.fullmatch(data)
    if result is None:
        if format_args:
            msg = msg.format(*format_args)
        raise LocalProtocolError(msg)
    return result.groupdict()


_T_Sentinel = TypeVar("_T_Sentinel", bound="Sentinel")


class Sentinel(type):
    def __new__(
        cls: Type[_T_Sentinel],
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwds: Any,
    ) -> _T_Sentinel:
        assert bases == (Sentinel,)
        created = super().__new__(cls, name, bases, namespace, **kwds)
        created.__class__ = created  # type: ignore[misc]
        return created

    def __repr__(self) -> str:
        return self.__name__


def bytesify(s: Union[bytes, bytearray, memoryview, int, str]) -> bytes:
    if type(s) is bytes:
        return s
    if isinstance(s, str):
        return s.encode("ascii")
    if isinstance(s, int):
        raise TypeError("expected bytes-like object, not int")
    return bytes(s)