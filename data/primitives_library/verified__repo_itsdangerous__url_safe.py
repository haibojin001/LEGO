from __future__ import annotations

import typing as t
import zlib

from ._json import _CompactJSON
from .encoding import base64_decode
from .encoding import base64_encode
from .exc import BadPayload
from .serializer import _PDataSerializer
from .serializer import Serializer
from .timed import TimedSerializer


class URLSafeSerializerMixin(Serializer[str]):
    """Add URL-safe base64 encoding and optional zlib compression."""

    default_serializer: _PDataSerializer[str] = _CompactJSON

    def load_payload(
        self,
        payload: bytes,
        *args: t.Any,
        serializer: t.Any | None = None,
        **kwargs: t.Any,
    ) -> t.Any:
        compressed = payload.startswith(b".")

        if compressed:
            payload = payload[1:]

        try:
            decoded = base64_decode(payload)
        except Exception as error:
            raise BadPayload(
                "Could not base64 decode the payload because of an exception",
                original_error=error,
            ) from error

        if compressed:
            try:
                decoded = zlib.decompress(decoded)
            except Exception as error:
                raise BadPayload(
                    "Could not zlib decompress the payload before decoding the payload",
                    original_error=error,
                ) from error

        return super().load_payload(decoded, *args, **kwargs)

    def dump_payload(self, obj: t.Any) -> bytes:
        payload = super().dump_payload(obj)
        compressed = zlib.compress(payload)

        if len(compressed) < len(payload) - 1:
            return b"." + base64_encode(compressed)

        return base64_encode(payload)


class URLSafeSerializer(URLSafeSerializerMixin, Serializer[str]):
    """A serializer producing URL-safe signed strings."""


class URLSafeTimedSerializer(URLSafeSerializerMixin, TimedSerializer[str]):
    """A timed serializer producing URL-safe signed strings."""