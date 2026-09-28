import re
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Union

from ._abnf import method, request_target
from ._headers import Headers, normalize_and_validate
from ._util import LocalProtocolError, bytesify, validate

__all__ = [
    "Event",
    "Request",
    "InformationalResponse",
    "Response",
    "Data",
    "EndOfMessage",
    "ConnectionClosed",
]

method_re = re.compile(method.encode("ascii"))
request_target_re = re.compile(request_target.encode("ascii"))


class Event(ABC):
    __slots__ = ()


@dataclass(init=False, frozen=True)
class Request(Event):
    __slots__ = ("method", "headers", "target", "http_version")

    method: bytes
    headers: Headers
    target: bytes
    http_version: bytes

    def __init__(
        self,
        *,
        method: Union[bytes, str],
        headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]],
        target: Union[bytes, str],
        http_version: Union[bytes, str] = b"1.1",
        _parsed: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(headers, Headers):
            checked_headers = headers
        else:
            checked_headers = normalize_and_validate(headers, _parsed=_parsed)
        object.__setattr__(self, "headers", checked_headers)

        if _parsed:
            actual_method = method
            actual_target = target
            actual_version = http_version
        else:
            actual_method = bytesify(method)
            actual_target = bytesify(target)
            actual_version = bytesify(http_version)

        object.__setattr__(self, "method", actual_method)
        object.__setattr__(self, "target", actual_target)
        object.__setattr__(self, "http_version", actual_version)

        hosts = 0
        for name, value in checked_headers:
            if name == b"host":
                hosts += 1

        if actual_version == b"1.1" and hosts == 0:
            raise LocalProtocolError("Missing mandatory Host: header")
        if hosts > 1:
            raise LocalProtocolError("Found multiple Host: headers")

        validate(method_re, actual_method, "Illegal method characters")
        validate(request_target_re, actual_target, "Illegal target characters")

    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class _ResponseBase(Event):
    __slots__ = ("headers", "http_version", "reason", "status_code")

    headers: Headers
    http_version: bytes
    reason: bytes
    status_code: int

    def __init__(
        self,
        *,
        headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]],
        status_code: int,
        http_version: Union[bytes, str] = b"1.1",
        reason: Union[bytes, str] = b"",
        _parsed: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(headers, Headers):
            checked_headers = headers
        else:
            checked_headers = normalize_and_validate(headers, _parsed=_parsed)
        object.__setattr__(self, "headers", checked_headers)

        if _parsed:
            actual_reason = reason
            actual_version = http_version
            actual_status = status_code
        else:
            actual_reason = bytesify(reason)
            actual_version = bytesify(http_version)
            if not isinstance(status_code, int):
                raise LocalProtocolError("status code must be integer")
            actual_status = int(status_code)

        object.__setattr__(self, "reason", actual_reason)
        object.__setattr__(self, "http_version", actual_version)
        object.__setattr__(self, "status_code", actual_status)

        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class InformationalResponse(_ResponseBase):
    def __post_init__(self) -> None:
        if not 100 <= self.status_code < 200:
            raise LocalProtocolError(
                "InformationalResponse status_code should be in range "
                "[100, 200), not {}".format(self.status_code)
            )

    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class Response(_ResponseBase):
    def __post_init__(self) -> None:
        if not 200 <= self.status_code < 1000:
            raise LocalProtocolError(
                "Response status_code should be in range [200, 1000), not {}".format(
                    self.status_code
                )
            )

    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class Data(Event):
    __slots__ = ("data", "chunk_start", "chunk_end")

    data: bytes
    chunk_start: bool
    chunk_end: bool

    def __init__(
        self, data: bytes, chunk_start: bool = False, chunk_end: bool = False
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "chunk_start", chunk_start)
        object.__setattr__(self, "chunk_end", chunk_end)

    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class EndOfMessage(Event):
    __slots__ = ("headers",)

    headers: Headers

    def __init__(
        self,
        headers: Union[
            Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]
        ] = None,
        _parsed: bool = False,
    ) -> None:
        if headers is None:
            headers = []

        if isinstance(headers, Headers):
            checked_headers = headers
        else:
            checked_headers = normalize_and_validate(headers, _parsed=_parsed)

        object.__setattr__(self, "headers", checked_headers)

    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class ConnectionClosed(Event):
    __slots__ = ()

    __hash__ = None  # type: ignore