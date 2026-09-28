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
    """Mixed in with a regular serializer it will attempt to zlib
    compress the string to make it shorter if necessary. It will also
    base64 encode the string so that it can safely be placed in a URL.
    """

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
            payload = base64_decode(payload)
        except Exception as error:
            raise BadPayload(
                "Could not base64 decode the payload because of an exception",
                original_error=error,
            ) from error

        if compressed:
            try:
                payload = zlib.decompress(payload)
            except Exception as error:
                raise BadPayload(
                    "Could not zlib decompress the payload before decoding the payload",
                    original_error=error,
                ) from error

        return super().load_payload(payload, *args, **kwargs)

    def dump_payload(self, obj: t.Any) -> bytes:
        payload = super().dump_payload(obj)
        packed = zlib.compress(payload)

        if len(packed) < len(payload) - 1:
            return b"." + base64_encode(packed)

        return base64_encode(payload)


class URLSafeSerializer(URLSafeSerializerMixin, Serializer[str]):
    """Works like :class:`.Serializer` but dumps and loads into a URL
    safe string consisting of the upper and lowercase character of the
    alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """


class URLSafeTimedSerializer(URLSafeSerializerMixin, TimedSerializer[str]):
    """Works like :class:`.TimedSerializer` but dumps and loads into a URL
    safe string consisting of the upper and lowercase character of the
    alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """