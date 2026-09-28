from __future__ import annotations

from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

from .events import (
    BytesMessage,
    CloseConnection,
    Event,
    Message,
    Ping,
    Pong,
    TextMessage,
)
from .frame_protocol import CloseReason, FrameProtocol, Opcode, ParseFailed
from .utilities import LocalProtocolError

if TYPE_CHECKING:
    from collections.abc import Generator

    from .extensions import Extension


class ConnectionState(Enum):
    """The lifecycle state of a WebSocket connection."""

    CONNECTING = 0
    OPEN = 1
    REMOTE_CLOSING = 2
    LOCAL_CLOSING = 3
    CLOSED = 4
    REJECTING = 5


class ConnectionType(Enum):
    """The role this endpoint has in a WebSocket connection."""

    CLIENT = 1
    SERVER = 2


CLIENT = ConnectionType.CLIENT
SERVER = ConnectionType.SERVER


class Connection:
    """A WebSocket framing connection."""

    def __init__(
        self,
        connection_type: ConnectionType,
        extensions: list[Extension] | None = None,
        trailing_data: bytes = b"",
    ) -> None:
        self.client = connection_type is ConnectionType.CLIENT
        self._events: deque[Event] = deque()
        self._proto = FrameProtocol(self.client, extensions or [])
        self._state = ConnectionState.OPEN
        self.receive_data(trailing_data)

    @property
    def state(self) -> ConnectionState:
        return self._state

    def send(self, event: Event) -> bytes:
        if isinstance(event, Message) and self._state is ConnectionState.OPEN:
            return self._proto.send_data(event.data, event.message_finished)

        if isinstance(event, Ping) and self._state is ConnectionState.OPEN:
            return self._proto.ping(event.payload)

        if isinstance(event, Pong) and self._state is ConnectionState.OPEN:
            return self._proto.pong(event.payload)

        if isinstance(event, CloseConnection) and self._state in (
            ConnectionState.OPEN,
            ConnectionState.REMOTE_CLOSING,
        ):
            result = self._proto.close(event.code, event.reason)
            if self._state is ConnectionState.REMOTE_CLOSING:
                self._state = ConnectionState.CLOSED
            else:
                self._state = ConnectionState.LOCAL_CLOSING
            return result

        raise LocalProtocolError(
            f"Event {event} cannot be sent in state {self.state}."
        )

    def receive_data(self, data: bytes | None) -> None:
        if data is None:
            self._events.append(
                CloseConnection(code=CloseReason.ABNORMAL_CLOSURE)
            )
            self._state = ConnectionState.CLOSED
            return

        if self._state in (
            ConnectionState.OPEN,
            ConnectionState.LOCAL_CLOSING,
        ):
            self._proto.receive_bytes(data)
            return

        if self._state is ConnectionState.CLOSED:
            raise LocalProtocolError("Connection already closed.")

    def events(self) -> Generator[Event, None, None]:
        while self._events:
            yield self._events.popleft()

        try:
            for frame in self._proto.received_frames():
                if frame.opcode is Opcode.PING:
                    assert frame.frame_finished
                    assert frame.message_finished
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield Ping(payload=frame.payload)
                    continue

                if frame.opcode is Opcode.PONG:
                    assert frame.frame_finished
                    assert frame.message_finished
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield Pong(payload=frame.payload)
                    continue

                if frame.opcode is Opcode.CLOSE:
                    assert isinstance(frame.payload, tuple)
                    close_code, close_reason = frame.payload
                    if self._state is ConnectionState.LOCAL_CLOSING:
                        self._state = ConnectionState.CLOSED
                    else:
                        self._state = ConnectionState.REMOTE_CLOSING
                    yield CloseConnection(code=close_code, reason=close_reason)
                    continue

                if frame.opcode is Opcode.TEXT:
                    assert isinstance(frame.payload, str)
                    yield TextMessage(
                        data=frame.payload,
                        frame_finished=frame.frame_finished,
                        message_finished=frame.message_finished,
                    )
                    continue

                if frame.opcode is Opcode.BINARY:
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield BytesMessage(
                        data=frame.payload,
                        frame_finished=frame.frame_finished,
                        message_finished=frame.message_finished,
                    )
        except ParseFailed as error:
            yield CloseConnection(code=error.code, reason=str(error))