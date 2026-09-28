import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union

from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
    CLIENT,
    CLOSED,
    DONE,
    IDLE,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
)
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate

__all__ = ["READERS"]


header_field_re = re.compile(header_field.encode("ascii"))
obs_fold_re = re.compile(rb"[ \t]+")
request_line_re = re.compile(request_line.encode("ascii"))
status_line_re = re.compile(status_line.encode("ascii"))
chunk_header_re = re.compile(chunk_header.encode("ascii"))


def _obsolete_line_fold(lines: Iterable[bytes]) -> Iterable[bytes]:
    previous: Optional[bytes] = None

    for current in lines:
        folding = obs_fold_re.match(current)
        if folding is None:
            if previous is not None:
                yield previous
            previous = current
            continue

        if previous is None:
            raise LocalProtocolError("continuation line at start of headers")

        if not isinstance(previous, bytearray):
            previous = bytearray(previous)
        previous.extend(b" ")
        previous.extend(current[folding.end() :])

    if previous is not None:
        yield previous


def _decode_header_lines(
    lines: Iterable[bytes],
) -> Iterable[Tuple[bytes, bytes]]:
    for line in _obsolete_line_fold(lines):
        match = validate(header_field_re, line, "illegal header line: {!r}", line)
        yield match["field_name"], match["field_value"]


def maybe_read_from_IDLE_client(buf: ReceiveBuffer) -> Optional[Request]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError("illegal request line")
        return None

    if len(lines) == 0:
        raise LocalProtocolError("no request line received")

    match = validate(
        request_line_re,
        lines[0],
        "illegal request line: {!r}",
        lines[0],
    )
    return Request(
        headers=list(_decode_header_lines(lines[1:])),
        _parsed=True,
        **match,
    )


def maybe_read_from_SEND_RESPONSE_server(
    buf: ReceiveBuffer,
) -> Union[InformationalResponse, Response, None]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError("illegal request line")
        return None

    if len(lines) == 0:
        raise LocalProtocolError("no response line received")

    match = validate(
        status_line_re,
        lines[0],
        "illegal status line: {!r}",
        lines[0],
    )

    version = match["http_version"]
    if version is None:
        version = b"1.1"

    reason = match["reason"]
    if reason is None:
        reason = b""

    code = int(match["status_code"])
    event_type: Union[Type[InformationalResponse], Type[Response]]
    if code < 200:
        event_type = InformationalResponse
    else:
        event_type = Response

    return event_type(
        headers=list(_decode_header_lines(lines[1:])),
        _parsed=True,
        status_code=code,
        reason=reason,
        http_version=version,
    )


class ContentLengthReader:
    def __init__(self, length: int) -> None:
        self._length = length
        self._remaining = length

    def __call__(self, buf: ReceiveBuffer) -> Union[Data, EndOfMessage, None]:
        if self._remaining == 0:
            return EndOfMessage()

        data = buf.maybe_extract_at_most(self._remaining)
        if data is None:
            return None

        self._remaining -= len(data)
        return Data(data=data)

    def read_eof(self) -> NoReturn:
        received = self._length - self._remaining
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(received {} bytes, expected {})".format(received, self._length)
        )


class ChunkedReader:
    def __init__(self) -> None:
        self._bytes_in_chunk = 0
        self._bytes_to_discard = b""
        self._reading_trailer = False

    def __call__(self, buf: ReceiveBuffer) -> Union[Data, EndOfMessage, None]:
        if self._reading_trailer:
            trailer_lines = buf.maybe_extract_lines()
            if trailer_lines is None:
                return None
            return EndOfMessage(headers=list(_decode_header_lines(trailer_lines)))

        if self._bytes_to_discard:
            discarded = buf.maybe_extract_at_most(len(self._bytes_to_discard))
            if discarded is None:
                return None

            expected = self._bytes_to_discard[: len(discarded)]
            if discarded != expected:
                raise LocalProtocolError(
                    f"malformed chunk footer: {discarded!r} "
                    f"(expected {self._bytes_to_discard!r})"
                )

            self._bytes_to_discard = self._bytes_to_discard[len(discarded) :]
            if self._bytes_to_discard:
                return None

        if self._bytes_in_chunk == 0:
            line = buf.maybe_extract_next_line()
            if line is None:
                return None

            match = validate(
                chunk_header_re,
                line,
                "illegal chunk header: {!r}",
                line,
            )
            self._bytes_in_chunk = int(match["chunk_size"], 16)

            if self._bytes_in_chunk == 0:
                self._reading_trailer = True
                return self(buf)

            is_chunk_start = True
        else:
            is_chunk_start = False

        data = buf.maybe_extract_at_most(self._bytes_in_chunk)
        if data is None:
            return None

        self._bytes_in_chunk -= len(data)
        is_chunk_end = self._bytes_in_chunk == 0
        if is_chunk_end:
            self._bytes_to_discard = b"\r\n"

        return Data(
            data=data,
            chunk_start=is_chunk_start,
            chunk_end=is_chunk_end,
        )

    def read_eof(self) -> NoReturn:
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )


class Http10Reader:
    def __call__(self, buf: ReceiveBuffer) -> Optional[Data]:
        data = buf.maybe_extract_at_most(999999999)
        if data is None:
            return None
        return Data(data=data)

    def read_eof(self) -> EndOfMessage:
        return EndOfMessage()


def expect_nothing(buf: ReceiveBuffer) -> None:
    if buf:
        raise LocalProtocolError("Got data when expecting EOF")
    return None


ReadersType = Dict[
    Union[Type[Sentinel], Tuple[Type[Sentinel], Type[Sentinel]]],
    Union[Callable[..., Any], Dict[str, Callable[..., Any]]],
]


READERS: ReadersType = {
    (CLIENT, IDLE): maybe_read_from_IDLE_client,
    (SERVER, IDLE): maybe_read_from_SEND_RESPONSE_server,
    (SERVER, SEND_RESPONSE): maybe_read_from_SEND_RESPONSE_server,
    (CLIENT, DONE): expect_nothing,
    (CLIENT, MUST_CLOSE): expect_nothing,
    (CLIENT, CLOSED): expect_nothing,
    (SERVER, DONE): expect_nothing,
    (SERVER, MUST_CLOSE): expect_nothing,
    (SERVER, CLOSED): expect_nothing,
    SEND_BODY: {
        "chunked": ChunkedReader,
        "content-length": ContentLengthReader,
        "http/1.0": Http10Reader,
    },
}