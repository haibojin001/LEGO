import array
import os
import struct
import sys
from threading import Lock
from typing import Any, Callable, Optional, Union

from ._exceptions import WebSocketPayloadException, WebSocketProtocolException
from ._utils import validate_utf8

try:
    from wsaccel.xormask import XorMaskerSimple

    def _mask(mask_value: array.array, data_value: array.array) -> bytes:
        return XorMaskerSimple(mask_value).process(data_value)

except ImportError:
    _byteorder = sys.byteorder

    def _mask(mask_value: array.array, data_value: array.array) -> bytes:
        size = len(data_value)
        source = int.from_bytes(data_value, _byteorder)
        repeated_mask = mask_value * (size // 4) + mask_value[: size % 4]
        mask = int.from_bytes(repeated_mask, _byteorder)
        return (source ^ mask).to_bytes(size, _byteorder)


__all__ = [
    "ABNF",
    "continuous_frame",
    "frame_buffer",
    "STATUS_NORMAL",
    "STATUS_GOING_AWAY",
    "STATUS_PROTOCOL_ERROR",
    "STATUS_UNSUPPORTED_DATA_TYPE",
    "STATUS_STATUS_NOT_AVAILABLE",
    "STATUS_ABNORMAL_CLOSED",
    "STATUS_INVALID_PAYLOAD",
    "STATUS_POLICY_VIOLATION",
    "STATUS_MESSAGE_TOO_BIG",
    "STATUS_INVALID_EXTENSION",
    "STATUS_UNEXPECTED_CONDITION",
    "STATUS_SERVICE_RESTART",
    "STATUS_TRY_AGAIN_LATER",
    "STATUS_BAD_GATEWAY",
    "STATUS_TLS_HANDSHAKE_ERROR",
]

STATUS_NORMAL = 1000
STATUS_GOING_AWAY = 1001
STATUS_PROTOCOL_ERROR = 1002
STATUS_UNSUPPORTED_DATA_TYPE = 1003
STATUS_STATUS_NOT_AVAILABLE = 1005
STATUS_ABNORMAL_CLOSED = 1006
STATUS_INVALID_PAYLOAD = 1007
STATUS_POLICY_VIOLATION = 1008
STATUS_MESSAGE_TOO_BIG = 1009
STATUS_INVALID_EXTENSION = 1010
STATUS_UNEXPECTED_CONDITION = 1011
STATUS_SERVICE_RESTART = 1012
STATUS_TRY_AGAIN_LATER = 1013
STATUS_BAD_GATEWAY = 1014
STATUS_TLS_HANDSHAKE_ERROR = 1015

VALID_CLOSE_STATUS = (
    STATUS_NORMAL,
    STATUS_GOING_AWAY,
    STATUS_PROTOCOL_ERROR,
    STATUS_UNSUPPORTED_DATA_TYPE,
    STATUS_INVALID_PAYLOAD,
    STATUS_POLICY_VIOLATION,
    STATUS_MESSAGE_TOO_BIG,
    STATUS_INVALID_EXTENSION,
    STATUS_UNEXPECTED_CONDITION,
    STATUS_SERVICE_RESTART,
    STATUS_TRY_AGAIN_LATER,
    STATUS_BAD_GATEWAY,
)


class ABNF:
    OPCODE_CONT = 0x0
    OPCODE_TEXT = 0x1
    OPCODE_BINARY = 0x2
    OPCODE_CLOSE = 0x8
    OPCODE_PING = 0x9
    OPCODE_PONG = 0xA

    OPCODES = (
        OPCODE_CONT,
        OPCODE_TEXT,
        OPCODE_BINARY,
        OPCODE_CLOSE,
        OPCODE_PING,
        OPCODE_PONG,
    )

    OPCODE_MAP = {
        OPCODE_CONT: "cont",
        OPCODE_TEXT: "text",
        OPCODE_BINARY: "binary",
        OPCODE_CLOSE: "close",
        OPCODE_PING: "ping",
        OPCODE_PONG: "pong",
    }

    LENGTH_7 = 0x7E
    LENGTH_16 = 1 << 16
    LENGTH_63 = 1 << 63

    def __init__(
        self,
        fin: int = 0,
        rsv1: int = 0,
        rsv2: int = 0,
        rsv3: int = 0,
        opcode: int = OPCODE_TEXT,
        mask_value: int = 1,
        data: Optional[Union[str, bytes]] = "",
    ) -> None:
        self.fin = fin
        self.rsv1 = rsv1
        self.rsv2 = rsv2
        self.rsv3 = rsv3
        self.opcode = opcode
        self.mask_value = mask_value
        self.data = "" if data is None else data
        self.get_mask_key: Callable[[int], bytes] = os.urandom

    def validate(self, skip_utf8_validation: bool = False) -> None:
        if self.rsv1 or self.rsv2 or self.rsv3:
            raise WebSocketProtocolException("rsv is not implemented, yet")

        if self.opcode not in self.OPCODES:
            raise WebSocketProtocolException(f"Invalid opcode {self.opcode!r}")

        if self.opcode == self.OPCODE_PING and not self.fin:
            raise WebSocketProtocolException("Invalid ping frame.")

        if self.opcode != self.OPCODE_CLOSE:
            return

        data_length = len(self.data)
        if data_length == 0:
            return
        if data_length == 1 or data_length >= self.LENGTH_7:
            raise WebSocketProtocolException("Invalid close frame.")

        if (
            data_length > 2
            and not skip_utf8_validation
            and not validate_utf8(self.data[2:])
        ):
            raise WebSocketProtocolException("Invalid close frame.")

        if isinstance(self.data, bytes):
            code_data = self.data[:2]
        else:
            code_data = self.data[:2].encode("utf-8")
        code = struct.unpack("!H", code_data)[0]
        if not self._is_valid_close_status(code):
            raise WebSocketProtocolException(f"Invalid close opcode {code!r}")

    @staticmethod
    def _is_valid_close_status(code: int) -> bool:
        return code in VALID_CLOSE_STATUS or 3000 <= code < 5000

    def __str__(self) -> str:
        value = self.data if isinstance(self.data, str) else repr(self.data)
        return f"fin={self.fin} opcode={self.opcode} data={value}"

    @staticmethod
    def create_frame(data: Union[bytes, str], opcode: int, fin: int = 1) -> "ABNF":
        if opcode == ABNF.OPCODE_TEXT and isinstance(data, str):
            data = data.encode("utf-8")
        return ABNF(fin, 0, 0, 0, opcode, 1, data)

    def format(self) -> bytes:
        flags = (self.fin, self.rsv1, self.rsv2, self.rsv3)
        if any(flag not in (0, 1) for flag in flags):
            raise ValueError("not 0 or 1")
        if self.opcode not in self.OPCODES:
            raise ValueError("Invalid OPCODE")

        data_length = len(self.data)
        if data_length >= self.LENGTH_63:
            raise ValueError("data is too long")

        first_byte = (
            (self.fin << 7)
            | (self.rsv1 << 6)
            | (self.rsv2 << 5)
            | (self.rsv3 << 4)
            | self.opcode
        )
        header = bytes((first_byte,))

        if data_length < self.LENGTH_7:
            header += bytes(((self.mask_value << 7) | data_length,))
        elif data_length < self.LENGTH_16:
            header += bytes(((self.mask_value << 7) | 0x7E,))
            header += struct.pack("!H", data_length)
        else:
            header += bytes(((self.mask_value << 7) | 0x7F,))
            header += struct.pack("!Q", data_length)

        if not self.mask_value:
            if isinstance(self.data, str):
                self.data = self.data.encode("utf-8")
            return header + self.data

        mask_key = self.get_mask_key(4)
        return header + self._get_masked(mask_key)

    def _get_masked(self, mask_key: Union[str, bytes]) -> bytes:
        masked = self.mask(mask_key, self.data)
        if isinstance(mask_key, str):
            mask_key = mask_key.encode("utf-8")
        return mask_key + masked

    @staticmethod
    def mask(mask_key: Union[str, bytes], data: Union[str, bytes]) -> bytes:
        if isinstance(mask_key, str):
            mask_key = mask_key.encode("utf-8")
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _mask(array.array("B", mask_key), array.array("B", data))


class continuous_frame:
    def __init__(self, fire_cont_frame: bool, skip_utf8_validation: bool) -> None:
        self.fire_cont_frame = fire_cont_frame
        self.skip_utf8_validation = skip_utf8_validation
        self.cont_data: Optional[int] = None
        self.cont_data_list: list = []

    def validate(self, frame: ABNF) -> None:
        if frame.opcode in (
            ABNF.OPCODE_CLOSE,
            ABNF.OPCODE_PING,
            ABNF.OPCODE_PONG,
        ):
            return

        if self.cont_data is None:
            if frame.opcode == ABNF.OPCODE_CONT:
                raise WebSocketProtocolException("Illegal frame")
            if not frame.fin:
                self.cont_data = frame.opcode
                self.cont_data_list.append(frame.data)
            return

        if frame.opcode != ABNF.OPCODE_CONT:
            raise WebSocketProtocolException("Illegal frame")
        self.cont_data_list.append(frame.data)

    def add(self, frame: ABNF) -> tuple:
        if frame.opcode in (
            ABNF.OPCODE_CLOSE,
            ABNF.OPCODE_PING,
            ABNF.OPCODE_PONG,
        ):
            return frame.opcode, frame.data

        if self.cont_data is None:
            if frame.opcode == ABNF.OPCODE_CONT:
                raise WebSocketProtocolException("Illegal frame")

            if frame.fin:
                return frame.opcode, frame.data

            self.cont_data = frame.opcode
            self.cont_data_list.append(frame.data)
            if self.fire_cont_frame:
                return frame.opcode, frame.data
            return None, None

        if frame.opcode != ABNF.OPCODE_CONT:
            raise WebSocketProtocolException("Illegal frame")

        self.cont_data_list.append(frame.data)

        if not frame.fin:
            if self.fire_cont_frame:
                return frame.opcode, frame.data
            return None, None

        opcode = self.cont_data
        data = b"".join(self.cont_data_list)
        self.cont_data = None
        self.cont_data_list = []

        if (
            opcode == ABNF.OPCODE_TEXT
            and not self.skip_utf8_validation
            and not validate_utf8(data)
        ):
            raise WebSocketPayloadException(f"cannot decode: {data!r}")

        return opcode, data


class frame_buffer:
    def __init__(
        self,
        recv_fn: Callable[[int], bytes],
        skip_utf8_validation: bool,
    ) -> None:
        self.recv = recv_fn
        self.skip_utf8_validation = skip_utf8_validation
        self.recv_buffer: list = []
        self.header: Optional[tuple] = None
        self.length: Optional[int] = None
        self.mask_value: Optional[bytes] = None
        self.has_mask: Optional[int] = None
        self.lock = Lock()

    def clear(self) -> None:
        self.header = None
        self.length = None
        self.mask_value = None
        self.has_mask = None

    def recv_frame(self) -> ABNF:
        with self.lock:
            try:
                self.recv_header()
                self.recv_length()
                self.recv_mask()
                return self.recv_data()
            finally:
                self.clear()

    def recv_header(self) -> None:
        header = self.recv_strict(2)
        first, second = header[0], header[1]

        fin = (first >> 7) & 1
        rsv1 = (first >> 6) & 1
        rsv2 = (first >> 5) & 1
        rsv3 = (first >> 4) & 1
        opcode = first & 0x0F

        self.header = (fin, rsv1, rsv2, rsv3, opcode)
        self.has_mask = (second >> 7) & 1
        self.length = second & 0x7F

    def recv_length(self) -> None:
        if self.length == ABNF.LENGTH_7:
            self.length = struct.unpack("!H", self.recv_strict(2))[0]
        elif self.length == 0x7F:
            self.length = struct.unpack("!Q", self.recv_strict(8))[0]
            if self.length >= ABNF.LENGTH_63:
                raise WebSocketProtocolException("Invalid frame length")

    def recv_mask(self) -> None:
        if self.has_mask:
            self.mask_value = self.recv_strict(4)

    def recv_data(self) -> ABNF:
        if self.header is None or self.length is None:
            raise WebSocketProtocolException("Invalid frame header")

        payload = self.recv_strict(self.length)
        if self.has_mask:
            payload = ABNF.mask(self.mask_value, payload)

        frame = ABNF(*self.header, self.has_mask, payload)
        frame.validate(self.skip_utf8_validation)
        return frame

    def recv_strict(self, bufsize: int) -> bytes:
        data = b""
        while len(data) < bufsize:
            if not self.recv_buffer:
                self.recv_buffer.append(self.recv(bufsize - len(data)))
            data += self.recv_buffer.pop(0)

        if len(data) > bufsize:
            self.recv_buffer.append(data[bufsize:])
            data = data[:bufsize]

        return data