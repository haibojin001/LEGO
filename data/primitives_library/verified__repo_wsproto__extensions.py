from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import NamedTuple, Optional

from . import frame_protocol as _frame_protocol
from .frame_protocol import CloseReason, FrameProtocol


FrameDecoder = getattr(_frame_protocol, "FrameDecoder", FrameProtocol)


class _FallbackOpcode(IntEnum):
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA

    def iscontrol(self) -> bool:
        return self.value >= 0x8


class _FallbackRsvBits(NamedTuple):
    rsv1: bool
    rsv2: bool
    rsv3: bool


Opcode = getattr(_frame_protocol, "Opcode", _FallbackOpcode)
RsvBits = getattr(_frame_protocol, "RsvBits", _FallbackRsvBits)


def _opcode_value(opcode: object) -> int:
    try:
        return int(opcode)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = getattr(opcode, "value", None)
        if value is None:
            return -1
        return int(value)


def _is_control_opcode(opcode: object) -> bool:
    iscontrol = getattr(opcode, "iscontrol", None)
    if callable(iscontrol):
        return bool(iscontrol())
    return _opcode_value(opcode) >= 0x8


class Extension(ABC):
    name: str

    def enabled(self) -> bool:
        return False

    @abstractmethod
    def offer(self) -> bool | str:
        pass

    def accept(self, offer: str) -> bool | str | None:
        pass

    def finalize(self, offer: str) -> None:
        pass

    def frame_inbound_header(
        self,
        proto: FrameDecoder | FrameProtocol,
        opcode: Opcode,
        rsv: RsvBits,
        payload_length: int,
    ) -> CloseReason | RsvBits:
        return RsvBits(False, False, False)

    def frame_inbound_payload_data(
        self,
        proto: FrameDecoder | FrameProtocol,
        data: bytes,
    ) -> bytes | CloseReason:
        return data

    def frame_inbound_complete(
        self,
        proto: FrameDecoder | FrameProtocol,
        fin: bool,
    ) -> bytes | CloseReason | None:
        pass

    def frame_outbound(
        self,
        proto: FrameDecoder | FrameProtocol,
        opcode: Opcode,
        rsv: RsvBits,
        data: bytes,
        fin: bool,
    ) -> tuple[RsvBits, bytes]:
        return rsv, data


class PerMessageDeflate(Extension):
    name = "permessage-deflate"

    DEFAULT_CLIENT_MAX_WINDOW_BITS = 15
    DEFAULT_SERVER_MAX_WINDOW_BITS = 15

    def __init__(
        self,
        client_no_context_takeover: bool = False,
        client_max_window_bits: int | None = None,
        server_no_context_takeover: bool = False,
        server_max_window_bits: int | None = None,
    ) -> None:
        self.client_no_context_takeover = client_no_context_takeover
        self.server_no_context_takeover = server_no_context_takeover
        self._client_max_window_bits = self.DEFAULT_CLIENT_MAX_WINDOW_BITS
        self._server_max_window_bits = self.DEFAULT_SERVER_MAX_WINDOW_BITS

        if client_max_window_bits is not None:
            self.client_max_window_bits = client_max_window_bits
        if server_max_window_bits is not None:
            self.server_max_window_bits = server_max_window_bits

        self._compressor: Optional[zlib._Compress] = None
        self._decompressor: Optional[zlib._Decompress] = None
        self._inbound_is_compressible: bool | None = None
        self._inbound_compressed: bool | None = None
        self._enabled = False

    @property
    def client_max_window_bits(self) -> int:
        return self._client_max_window_bits

    @client_max_window_bits.setter
    def client_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._client_max_window_bits = value

    @property
    def server_max_window_bits(self) -> int:
        return self._server_max_window_bits

    @server_max_window_bits.setter
    def server_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._server_max_window_bits = value

    def _compressible_opcode(self, opcode: Opcode) -> bool:
        return _opcode_value(opcode) in (
            _opcode_value(Opcode.TEXT),
            _opcode_value(Opcode.BINARY),
            _opcode_value(Opcode.CONTINUATION),
        )

    def enabled(self) -> bool:
        return self._enabled

    def offer(self) -> bool | str:
        parameters = [
            f"client_max_window_bits={self.client_max_window_bits}",
            f"server_max_window_bits={self.server_max_window_bits}",
        ]

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")

        return "; ".join(parameters)

    def finalize(self, offer: str) -> None:
        bits = [bit.strip() for bit in offer.split(";")]

        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                self.client_max_window_bits = int(bit.split("=", 1)[1].strip())
            elif bit.startswith("server_max_window_bits"):
                self.server_max_window_bits = int(bit.split("=", 1)[1].strip())

        self._enabled = True

    def _parse_params(self, params: str) -> tuple[int | None, int | None]:
        client_max_window_bits = None
        server_max_window_bits = None

        bits = [bit.strip() for bit in params.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                if "=" in bit:
                    client_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    client_max_window_bits = self.client_max_window_bits
            elif bit.startswith("server_max_window_bits"):
                if "=" in bit:
                    server_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    server_max_window_bits = self.server_max_window_bits

        return client_max_window_bits, server_max_window_bits

    def accept(self, offer: str) -> bool | str | None:
        client_max_window_bits, server_max_window_bits = self._parse_params(offer)
        parameters = []

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")

        try:
            if client_max_window_bits is not None:
                parameters.append(
                    f"client_max_window_bits={client_max_window_bits}"
                )
                self.client_max_window_bits = client_max_window_bits

            if server_max_window_bits is not None:
                parameters.append(
                    f"server_max_window_bits={server_max_window_bits}"
                )
                self.server_max_window_bits = server_max_window_bits
        except ValueError:
            return None

        self._enabled = True
        return "; ".join(parameters)

    def frame_inbound_header(
        self,
        proto: FrameDecoder | FrameProtocol,
        opcode: Opcode,
        rsv: RsvBits,
        payload_length: int,
    ) -> CloseReason | RsvBits:
        if rsv.rsv1 and _is_control_opcode(opcode):
            return CloseReason.PROTOCOL_ERROR
        if rsv.rsv1 and _opcode_value(opcode) == _opcode_value(Opcode.CONTINUATION):
            return CloseReason.PROTOCOL_ERROR

        self._inbound_is_compressible = self._compressible_opcode(opcode)

        if self._inbound_compressed is None:
            self._inbound_compressed = rsv.rsv1

            if self._inbound_compressed:
                assert self._inbound_is_compressible

                if proto.client:
                    window_bits = self.server_max_window_bits
                else:
                    window_bits = self.client_max_window_bits

                if self._decompressor is None:
                    self._decompressor = zlib.decompressobj(-int(window_bits))

        return RsvBits(True, False, False)

    def frame_inbound_payload_data(
        self,
        proto: FrameDecoder | FrameProtocol,
        data: bytes,
    ) -> bytes | CloseReason:
        if not self._inbound_compressed or not self._inbound_is_compressible:
            return data

        assert self._decompressor is not None

        try:
            return self._decompressor.decompress(bytes(data))
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

    def frame_inbound_complete(
        self,
        proto: FrameDecoder | FrameProtocol,
        fin: bool,
    ) -> bytes | CloseReason | None:
        if not fin:
            return None

        if not self._inbound_is_compressible:
            self._inbound_compressed = None
            return None

        if not self._inbound_compressed:
            self._inbound_compressed = None
            return None

        assert self._decompressor is not None

        try:
            data = self._decompressor.decompress(b"\x00\x00\xff\xff")
            data += self._decompressor.flush()
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

        if proto.client:
            no_context_takeover = self.server_no_context_takeover
        else:
            no_context_takeover = self.client_no_context_takeover

        if no_context_takeover:
            self._decompressor = None

        self._inbound_compressed = None
        return data

    def frame_outbound(
        self,
        proto: FrameDecoder | FrameProtocol,
        opcode: Opcode,
        rsv: RsvBits,
        data: bytes,
        fin: bool,
    ) -> tuple[RsvBits, bytes]:
        if not self._compressible_opcode(opcode):
            return rsv, data

        if _opcode_value(opcode) == _opcode_value(Opcode.CONTINUATION):
            assert self._compressor is not None
        else:
            if proto.client:
                window_bits = self.client_max_window_bits
            else:
                window_bits = self.server_max_window_bits

            self._compressor = zlib.compressobj(
                zlib.Z_DEFAULT_COMPRESSION,
                zlib.DEFLATED,
                -int(window_bits),
            )

        assert self._compressor is not None
        compressed = self._compressor.compress(data)

        if fin:
            compressed += self._compressor.flush(zlib.Z_SYNC_FLUSH)
            compressed = compressed[:-4]

            if proto.client:
                no_context_takeover = self.client_no_context_takeover
            else:
                no_context_takeover = self.server_no_context_takeover

            if no_context_takeover:
                self._compressor = None

        return RsvBits(True, rsv.rsv2, rsv.rsv3), compressed