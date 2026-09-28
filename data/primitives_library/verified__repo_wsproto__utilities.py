from __future__ import annotations

import base64
import hashlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from h11._headers import Headers as H11Headers

    from .events import Event
    from .typing import Headers


ACCEPT_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class ProtocolError(Exception):
    pass


class LocalProtocolError(ProtocolError):
    """An error caused by an invalid local websocket operation."""


class RemoteProtocolError(ProtocolError):
    """An error caused by invalid data or behavior from the remote peer."""

    def __init__(self, message: str, event_hint: Event) -> None:
        self.event_hint = event_hint
        Exception.__init__(self, message)


def normed_header_dict(h11_headers: Headers | H11Headers) -> dict[bytes, bytes]:
    collected: dict[bytes, list[bytes]] = {}
    for header_name, header_value in h11_headers:
        if header_name not in collected:
            collected[header_name] = []
        collected[header_name].append(header_value)
    return {
        header_name: b", ".join(header_values)
        for header_name, header_values in collected.items()
    }


def split_comma_header(value: bytes) -> list[str]:
    pieces = value.split(b",")
    return [item.decode("ascii").strip() for item in pieces]


def generate_nonce() -> bytes:
    random_bytes = os.urandom(16)
    return base64.b64encode(random_bytes)


def generate_accept_token(token: bytes) -> bytes:
    digest = hashlib.sha1(token + ACCEPT_GUID).digest()
    return base64.b64encode(digest)