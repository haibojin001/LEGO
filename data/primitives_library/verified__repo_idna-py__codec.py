from __future__ import annotations

import codecs
from typing import Any

from .core import IDNAError, _max_domain_length, _unicode_dots_re, alabel, decode, encode, ulabel


class Codec(codecs.Codec):
    """Codec implementation for IDNA 2008 domain names."""

    def encode(
        self, data: str, errors: str = "strict"
    ) -> tuple[bytes, int]:
        if errors != "strict":
            raise IDNAError(
                f'Unsupported error handling "{errors}"',
                code="unsupported_errors",
            )
        if data == "":
            return b"", 0
        return encode(data), len(data)

    def decode(
        self, data: bytes, errors: str = "strict"
    ) -> tuple[str, int]:
        if errors != "strict":
            raise IDNAError(
                f'Unsupported error handling "{errors}"',
                code="unsupported_errors",
            )
        if not data:
            return "", 0
        return decode(data), len(data)


class IncrementalEncoder(codecs.BufferedIncrementalEncoder):
    """Buffered encoder which emits complete IDNA labels."""

    def __init__(self, errors: str = "strict") -> None:
        super().__init__(errors)
        self._emitted = 0
        self._trailing_dot = False

    def reset(self) -> None:
        super().reset()
        self._emitted = 0
        self._trailing_dot = False

    def getstate(self) -> Any:
        if not self.buffer and not self._emitted:
            return 0
        return self.buffer, self._emitted, self._trailing_dot

    def setstate(self, state: Any) -> None:
        if state:
            self.buffer, self._emitted, self._trailing_dot = state
        else:
            self.reset()

    def _buffer_encode(
        self, data: str, errors: str, final: bool
    ) -> tuple[bytes, int]:
        if errors != "strict":
            raise IDNAError(
                f'Unsupported error handling "{errors}"',
                code="unsupported_errors",
            )

        output = b""
        consumed = 0

        if data:
            parts = _unicode_dots_re.split(data)
            dot = b""

            if parts:
                if parts[-1] == "":
                    parts.pop()
                    dot = b"."
                elif not final:
                    parts.pop()
                    if parts:
                        dot = b"."

            encoded_parts: list[bytes] = []
            for part in parts:
                encoded_parts.append(alabel(part))
                if consumed:
                    consumed += 1
                consumed += len(part)

            output = b".".join(encoded_parts) + dot
            consumed += len(dot)

        self._emitted += len(output)
        if output:
            self._trailing_dot = output.endswith(b".")

        allowance = 1 if (self._trailing_dot or not final) else 0
        if self._emitted > _max_domain_length + allowance:
            raise IDNAError("Domain too long", code="domain_too_long")

        return output, consumed


class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    """Buffered decoder which emits complete IDNA labels."""

    def __init__(self, errors: str = "strict") -> None:
        super().__init__(errors)
        self._consumed = 0

    def reset(self) -> None:
        super().reset()
        self._consumed = 0

    def getstate(self) -> tuple[bytes, int]:
        return self.buffer, self._consumed

    def setstate(self, state: tuple[bytes, int]) -> None:
        self.buffer, self._consumed = state

    def _buffer_decode(
        self, data: Any, errors: str, final: bool
    ) -> tuple[str, int]:
        if errors != "strict":
            raise IDNAError(
                f'Unsupported error handling "{errors}"',
                code="unsupported_errors",
            )

        if not data:
            return "", 0

        if not isinstance(data, str):
            try:
                data = str(data, "ascii")
            except UnicodeDecodeError as exc:
                raise IDNAError(
                    "Invalid ASCII in A-label",
                    code="invalid_ascii",
                ) from exc

        if self._consumed + len(data) > _max_domain_length + 1:
            raise IDNAError("Domain too long", code="domain_too_long")

        parts = _unicode_dots_re.split(data)
        dot = ""

        if parts:
            if parts[-1] == "":
                parts.pop()
                dot = "."
            elif not final:
                parts.pop()
                if parts:
                    dot = "."

        decoded_parts: list[str] = []
        consumed = 0
        for part in parts:
            decoded_parts.append(ulabel(part))
            if consumed:
                consumed += 1
            consumed += len(part)

        output = ".".join(decoded_parts) + dot
        consumed += len(dot)
        self._consumed += consumed

        return output, consumed


class StreamWriter(Codec, codecs.StreamWriter):
    pass


class StreamReader(Codec, codecs.StreamReader):
    pass


def search_function(name: str) -> codecs.CodecInfo | None:
    if name != "idna2008":
        return None

    codec = Codec()
    return codecs.CodecInfo(
        name=name,
        encode=codec.encode,
        decode=codec.decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )


codecs.register(search_function)