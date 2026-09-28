from __future__ import annotations

import contextlib
import os
import struct
from codecs import IncrementalDecoder, getincrementaldecoder
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Generator

    from .extensions import Extension


_XOR_TABLE = [bytes(value ^ key for value in range(256)) for key in range(256)]


class XorMaskerSimple:
    def __init__(self, masking_key: bytearray | bytes) -> None:
        self._masking_key = masking_key

    def process(self, data: bytearray) -> bytearray:
        data = bytearray(data)
        if not data:
            return data

        first, second, third, fourth = (
            _XOR_TABLE[value] for value in self._masking_key
        )
        data[::4] = data[::4].translate(first)
        data[1::4] = data[1::4].translate(second)
        data[2::4] = data[2::4].translate(third)
        data[3::4] = data[3::4].translate(fourth)

        offset = len(data) % 4
        self._masking_key = self._masking_key[offset:] + self._masking_key[:offset]
        return data


class XorMaskerNull:
    def process(self, data: bytearray) -> bytearray:
        return data


PAYLOAD_LENGTH_TWO_BYTE = 126
PAYLOAD_LENGTH_EIGHT_BYTE = 127
MAX_PAYLOAD_NORMAL = 125
MAX_PAYLOAD_TWO_BYTE = 2**16 - 1
MAX_PAYLOAD_EIGHT_BYTE = 2**64 - 1
MAX_FRAME_PAYLOAD = MAX_PAYLOAD_EIGHT_BYTE

MASK_MASK = 0x80
PAYLOAD_LEN_MASK = 0x7F

FIN_MASK = 0x80
RSV1_MASK = 0x40
RSV2_MASK = 0x20
RSV3_MASK = 0x10
OPCODE_MASK = 0x0F


class Opcode(IntEnum):
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA

    def iscontrol(self) -> bool:
        return bool(self & 0x08)


class CloseReason(IntEnum):
    NORMAL_CLOSURE = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    NO_STATUS_RCVD = 1005
    ABNORMAL_CLOSURE = 1006
    INVALID_FRAME_PAYLOAD_DATA = 1007
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    MANDATORY_EXT = 1010
    INTERNAL_ERROR = 1011
    SERVICE_RESTART = 1012
    TRY_AGAIN_LATER = 1013
    TLS_HANDSHAKE_FAILED = 1015


LOCAL_ONLY_CLOSE_REASONS = (
    CloseReason.NO_STATUS_RCVD,
    CloseReason.ABNORMAL_CLOSURE,
    CloseReason.TLS_HANDSHAKE_FAILED,
)

MIN_CLOSE_REASON = 1000
MIN_PROTOCOL_CLOSE_REASON = 1000
MAX_PROTOCOL_CLOSE_REASON = 2999
MIN_LIBRARY_CLOSE_REASON = 3000
MAX_LIBRARY_CLOSE_REASON = 3999
MIN_PRIVATE_CLOSE_REASON = 4000
MAX_PRIVATE_CLOSE_REASON = 4999
MAX_CLOSE_REASON = 4999

NULL_MASK = struct.pack("!I", 0)


class ParseFailed(Exception):
    def __init__(
        self,
        msg: str,
        code: CloseReason = CloseReason.PROTOCOL_ERROR,
    ) -> None:
        super().__init__(msg)
        self.code = code


class RsvBits(NamedTuple):
    rsv1: bool
    rsv2: bool
    rsv3: bool


class Header(NamedTuple):
    fin: bool
    rsv: RsvBits
    opcode: Opcode
    payload_len: int
    masking_key: bytes | None


class Frame(NamedTuple):
    opcode: Opcode
    payload: bytes | str | tuple[int, str]
    frame_finished: bool
    message_finished: bool


def _truncate_utf8(data: bytes, nbytes: int) -> bytes:
    if len(data) <= nbytes:
        return data
    shortened = data[:nbytes]
    return shortened.decode("utf-8", errors="ignore").encode("utf-8")


class Buffer:
    def __init__(self, initial_bytes: bytes | None = None) -> None:
        self.buffer = bytearray()
        self.bytes_used = 0
        if initial_bytes:
            self.feed(initial_bytes)

    def feed(self, new_bytes: bytes) -> None:
        self.buffer += new_bytes

    def consume_at_most(self, nbytes: int) -> bytearray:
        if not nbytes:
            return bytearray()
        result = self.buffer[self.bytes_used : self.bytes_used + nbytes]
        self.bytes_used += len(result)
        return result

    def consume_exactly(self, nbytes: int) -> bytearray | None:
        if len(self.buffer) - self.bytes_used < nbytes:
            return None
        return self.consume_at_most(nbytes)

    def commit(self) -> None:
        del self.buffer[: self.bytes_used]
        self.bytes_used = 0

    def rollback(self) -> None:
        self.bytes_used = 0

    def __len__(self) -> int:
        return len(self.buffer) - self.bytes_used


class FrameDecoder:
    def __init__(self) -> None:
        self.opcode: Opcode | None = None
        self.decoder: IncrementalDecoder | None = None

    def process_frame(self, frame: Frame) -> Frame:
        assert not frame.opcode.iscontrol()

        opcode = frame.opcode
        if opcode is Opcode.CONTINUATION:
            if self.opcode is None:
                raise ParseFailed("Received a continuation frame with no message in progress")
            opcode = self.opcode
        elif self.opcode is None:
            self.opcode = opcode
            if opcode is Opcode.TEXT:
                self.decoder = getincrementaldecoder("utf-8")()
        elif opcode is not self.opcode:
            raise ParseFailed("Received a new data frame while a message is in progress")

        payload: bytes | str | tuple[int, str] = frame.payload
        if opcode is Opcode.TEXT:
            assert self.decoder is not None
            try:
                payload = self.decoder.decode(
                    payload if isinstance(payload, bytes) else bytes(payload),
                    final=frame.message_finished,
                )
            except UnicodeDecodeError:
                self.opcode = None
                self.decoder = None
                raise ParseFailed(
                    "Invalid UTF-8 payload",
                    CloseReason.INVALID_FRAME_PAYLOAD_DATA,
                )

        result = Frame(opcode, payload, frame.frame_finished, frame.message_finished)

        if frame.message_finished:
            self.opcode = None
            self.decoder = None

        return result


MessageDecoder = FrameDecoder


def _valid_close_code(code: int) -> bool:
    if code < MIN_CLOSE_REASON or code > MAX_CLOSE_REASON:
        return False
    if code == 1004:
        return False
    with contextlib.suppress(ValueError):
        if CloseReason(code) in LOCAL_ONLY_CLOSE_REASONS:
            return False
    return True


class FrameProtocol:
    def __init__(
        self,
        client: bool,
        extensions: list[Extension] | None = None,
    ) -> None:
        self.client = client
        self.extensions = extensions or []
        self._buffer = Buffer()
        self._header: Header | None = None
        self._remaining_payload = 0
        self._masker: XorMaskerSimple | XorMaskerNull | None = None
        self._inbound_opcode: Opcode | None = None
        self._outbound_opcode: Opcode | None = None
        self._message_decoder = FrameDecoder()

    def receive_bytes(self, data: bytes | None) -> None:
        if data is not None:
            self._buffer.feed(data)

    def received_frames(self) -> Generator[Frame, None, None]:
        while True:
            if self._header is None:
                header = self._parse_header()
                if header is None:
                    return

                self._header = header
                self._remaining_payload = header.payload_len
                self._masker = (
                    XorMaskerSimple(header.masking_key)
                    if header.masking_key is not None
                    else XorMaskerNull()
                )

                if not self._remaining_payload:
                    yield self._finish_frame(bytearray())
                    continue

            if len(self._buffer) == 0:
                return

            assert self._header is not None
            assert self._masker is not None

            chunk = self._buffer.consume_at_most(self._remaining_payload)
            self._buffer.commit()
            self._remaining_payload -= len(chunk)
            payload = self._masker.process(chunk)

            for extension in self.extensions:
                payload = bytearray(extension.frame_inbound_payload_data(self, payload))

            if self._remaining_payload == 0:
                yield self._finish_frame(payload)
            else:
                frame = Frame(
                    self._header.opcode,
                    bytes(payload),
                    False,
                    False,
                )
                if frame.opcode.iscontrol():
                    yield frame
                else:
                    yield self._message_decoder.process_frame(frame)

    def send_data(self, payload: bytes | str, fin: bool = True) -> bytes:
        if isinstance(payload, str):
            opcode = Opcode.TEXT
            encoded_payload = payload.encode("utf-8")
        else:
            opcode = Opcode.BINARY
            encoded_payload = bytes(payload)

        if self._outbound_opcode is not None:
            opcode = Opcode.CONTINUATION
            if fin:
                self._outbound_opcode = None
        elif not fin:
            self._outbound_opcode = opcode

        return self._serialize_frame(opcode, encoded_payload, fin)

    def send_ping(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PING, payload, True)

    def send_pong(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PONG, payload, True)

    def send_close(
        self,
        code: int | CloseReason | None = None,
        reason: str | None = None,
    ) -> bytes:
        if code is None:
            if reason is None:
                payload = b""
            else:
                code = CloseReason.NORMAL_CLOSURE
                payload = struct.pack("!H", int(code)) + _truncate_utf8(
                    reason.encode("utf-8"), 123
                )
        else:
            code = int(code)
            if not _valid_close_code(code):
                raise ValueError("Invalid close code")
            reason_bytes = b"" if reason is None else reason.encode("utf-8")
            payload = struct.pack("!H", code) + _truncate_utf8(reason_bytes, 123)

        return self._serialize_frame(Opcode.CLOSE, payload, True)

    def _parse_header(self) -> Header | None:
        initial = self._buffer.consume_exactly(2)
        if initial is None:
            return None

        first, second = initial
        payload_len = second & PAYLOAD_LEN_MASK
        extra_len = 0
        if payload_len == PAYLOAD_LENGTH_TWO_BYTE:
            extra_len = 2
        elif payload_len == PAYLOAD_LENGTH_EIGHT_BYTE:
            extra_len = 8

        masked = bool(second & MASK_MASK)
        total_remaining = extra_len + (4 if masked else 0)
        remainder = self._buffer.consume_exactly(total_remaining)
        if remainder is None:
            self._buffer.rollback()
            return None

        try:
            opcode = Opcode(first & OPCODE_MASK)
        except ValueError:
            self._buffer.rollback()
            raise ParseFailed("Invalid opcode")

        fin = bool(first & FIN_MASK)
        rsv = RsvBits(
            bool(first & RSV1_MASK),
            bool(first & RSV2_MASK),
            bool(first & RSV3_MASK),
        )

        position = 0
        if extra_len == 2:
            payload_len = struct.unpack("!H", remainder[:2])[0]
            position = 2
            if payload_len < PAYLOAD_LENGTH_TWO_BYTE:
                self._buffer.rollback()
                raise ParseFailed("Invalid payload length")
        elif extra_len == 8:
            payload_len = struct.unpack("!Q", remainder[:8])[0]
            position = 8
            if payload_len & (1 << 63):
                self._buffer.rollback()
                raise ParseFailed("Invalid payload length")
            if payload_len <= MAX_PAYLOAD_TWO_BYTE:
                self._buffer.rollback()
                raise ParseFailed("Invalid payload length")

        masking_key = bytes(remainder[position : position + 4]) if masked else None

        if self.client and masked:
            self._buffer.rollback()
            raise ParseFailed("Server sent a masked frame")
        if not self.client and not masked:
            self._buffer.rollback()
            raise ParseFailed("Client sent an unmasked frame")

        if opcode.iscontrol():
            if not fin:
                self._buffer.rollback()
                raise ParseFailed("Received fragmented control frame")
            if payload_len > MAX_PAYLOAD_NORMAL:
                self._buffer.rollback()
                raise ParseFailed("Control frame payload too long")

        if opcode is Opcode.CONTINUATION:
            if self._inbound_opcode is None:
                self._buffer.rollback()
                raise ParseFailed("Received a continuation frame with no message in progress")
            if fin:
                self._inbound_opcode = None
        elif not opcode.iscontrol():
            if self._inbound_opcode is not None:
                self._buffer.rollback()
                raise ParseFailed("Received a new data frame while a message is in progress")
            if not fin:
                self._inbound_opcode = opcode

        for extension in self.extensions:
            rsv = extension.frame_inbound_header(self, opcode, rsv, payload_len)

        if rsv.rsv1 or rsv.rsv2 or rsv.rsv3:
            self._buffer.rollback()
            raise ParseFailed("Reserved bit set unexpectedly")

        self._buffer.commit()
        return Header(fin, rsv, opcode, payload_len, masking_key)

    def _finish_frame(self, payload: bytearray) -> Frame:
        assert self._header is not None

        for extension in self.extensions:
            extra = extension.frame_inbound_complete(self, self._header.fin)
            if extra:
                payload += extra

        header = self._header
        self._header = None
        self._remaining_payload = 0
        self._masker = None

        frame = Frame(header.opcode, bytes(payload), True, header.fin)

        if frame.opcode is Opcode.CLOSE:
            frame = Frame(
                frame.opcode,
                self._parse_close(frame.payload),
                frame.frame_finished,
                frame.message_finished,
            )

        if frame.opcode.iscontrol():
            return frame

        return self._message_decoder.process_frame(frame)

    def _parse_close(self, payload: bytes | str | tuple[int, str]) -> tuple[int, str]:
        assert isinstance(payload, bytes)

        if not payload:
            return (int(CloseReason.NO_STATUS_RCVD), "")

        if len(payload) == 1:
            raise ParseFailed("Close frame payload must be at least 2 bytes")

        code = struct.unpack("!H", payload[:2])[0]
        if not _valid_close_code(code):
            raise ParseFailed("Invalid close code")

        try:
            reason = payload[2:].decode("utf-8")
        except UnicodeDecodeError:
            raise ParseFailed(
                "Invalid UTF-8 payload",
                CloseReason.INVALID_FRAME_PAYLOAD_DATA,
            )

        return (code, reason)

    def _serialize_frame(self, opcode: Opcode, payload: bytes, fin: bool) -> bytes:
        payload = bytes(payload)

        if opcode.iscontrol():
            if not fin:
                raise ValueError("Control frames must not be fragmented")
            if len(payload) > MAX_PAYLOAD_NORMAL:
                raise ValueError("Control frame payload too long")

        rsv = RsvBits(False, False, False)
        for extension in self.extensions:
            rsv, payload = extension.frame_outbound(self, opcode, rsv, payload, fin)
            payload = bytes(payload)

        first = int(opcode)
        if fin:
            first |= FIN_MASK
        if rsv.rsv1:
            first |= RSV1_MASK
        if rsv.rsv2:
            first |= RSV2_MASK
        if rsv.rsv3:
            first |= RSV3_MASK

        payload_len = len(payload)
        mask_bit = MASK_MASK if self.client else 0

        if payload_len <= MAX_PAYLOAD_NORMAL:
            header = struct.pack("!BB", first, mask_bit | payload_len)
        elif payload_len <= MAX_PAYLOAD_TWO_BYTE:
            header = struct.pack(
                "!BBH",
                first,
                mask_bit | PAYLOAD_LENGTH_TWO_BYTE,
                payload_len,
            )
        elif payload_len <= MAX_PAYLOAD_EIGHT_BYTE:
            header = struct.pack(
                "!BBQ",
                first,
                mask_bit | PAYLOAD_LENGTH_EIGHT_BYTE,
                payload_len,
            )
        else:
            raise ValueError("Payload too large")

        if self.client:
            masking_key = os.urandom(4)
            masked_payload = XorMaskerSimple(masking_key).process(bytearray(payload))
            return header + masking_key + bytes(masked_payload)

        return header + payload