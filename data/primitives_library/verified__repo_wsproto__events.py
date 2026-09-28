from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .extensions import Extension
    from .typing import Headers


class Event(ABC):
    """Base class for wsproto events."""


@dataclass(frozen=True)
class Request(Event):
    """A WebSocket HTTP upgrade request."""

    host: str
    target: str
    extensions: Sequence[Extension] | Sequence[str] = field(default_factory=list)
    extra_headers: Headers = field(default_factory=list)
    subprotocols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AcceptConnection(Event):
    """Acceptance of a WebSocket upgrade request."""

    subprotocol: str | None = None
    extensions: list[Extension] = field(default_factory=list)
    extra_headers: Headers = field(default_factory=list)


@dataclass(frozen=True)
class RejectConnection(Event):
    """Rejection of a WebSocket upgrade request."""

    status_code: int = 400
    headers: Headers = field(default_factory=list)
    has_body: bool = False


@dataclass(frozen=True)
class RejectData(Event):
    """A chunk of an HTTP rejection response body."""

    data: bytes
    body_finished: bool = True


@dataclass(frozen=True)
class CloseConnection(Event):
    """A WebSocket close frame."""

    code: int
    reason: str | None = None

    def response(self) -> CloseConnection:
        """Create the corresponding close response event."""
        return CloseConnection(code=self.code, reason=self.reason)


T = TypeVar("T", bytes | bytearray, str)


@dataclass(frozen=True)
class Message(Event, Generic[T]):
    """A chunk of WebSocket message data."""

    data: T
    frame_finished: bool = True
    message_finished: bool = True


@dataclass(frozen=True)
class TextMessage(Message[str]):
    """A text WebSocket message chunk."""


@dataclass(frozen=True)
class BytesMessage(Message[bytearray | bytes]):
    """A binary WebSocket message chunk."""


@dataclass(frozen=True)
class Ping(Event):
    """A WebSocket ping frame."""

    payload: bytes = b""

    def response(self) -> Pong:
        """Create the corresponding pong response event."""
        return Pong(payload=self.payload)


@dataclass(frozen=True)
class Pong(Event):
    """A WebSocket pong frame."""

    payload: bytes = b""