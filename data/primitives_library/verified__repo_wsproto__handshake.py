from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import h11

from .connection import Connection, ConnectionState, ConnectionType
from .events import AcceptConnection, Event, RejectConnection, RejectData, Request
from .utilities import (
    LocalProtocolError,
    RemoteProtocolError,
    generate_accept_token,
    generate_nonce,
    normed_header_dict,
    split_comma_header,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    from .extensions import Extension
    from .typing import Headers


WEBSOCKET_VERSION = b"13"
WEBSOCKET_UPGRADE = b"websocket"


def _extension_parts(value: str) -> tuple[str, str]:
    name, separator, parameters = value.partition(";")
    if separator:
        return name.strip(), parameters.strip()
    return name.strip(), ""


def _extension_header_value(extension: Extension, value: object) -> bytes | None:
    if value is False or value is None:
        return None
    name = extension.name
    if value is True or value == "":
        return name.encode("ascii")
    return (name + "; " + str(value)).encode("ascii")


class H11Handshake:
    """A Handshake implementation for HTTP/1.1 connections."""

    def __init__(self, connection_type: ConnectionType) -> None:
        self.client = connection_type is ConnectionType.CLIENT
        self._state = ConnectionState.CONNECTING
        self._h11_connection = h11.Connection(
            h11.CLIENT if self.client else h11.SERVER,
        )
        self._connection: Connection | None = None
        self._events: deque[Event] = deque()
        self._initiating_request: Request | None = None
        self._nonce: bytes | None = None

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def connection(self) -> Connection | None:
        return self._connection

    def initiate_upgrade_connection(
        self, headers: Headers, path: bytes | str,
    ) -> None:
        if self.client:
            raise LocalProtocolError(
                "Cannot initiate an upgrade connection when acting as the client",
            )

        request = h11.Request(method=b"GET", target=path, headers=headers)
        client_connection = h11.Connection(h11.CLIENT)
        self.receive_data(cast(bytes, client_connection.send(request)))

    def send(self, event: Event) -> bytes:
        if isinstance(event, Request):
            return self._initiate_connection(event)
        if isinstance(event, AcceptConnection):
            return self._accept(event)
        if isinstance(event, RejectConnection):
            return self._reject(event)
        if isinstance(event, RejectData):
            return self._send_reject_data(event)
        raise LocalProtocolError(
            f"Event {event} cannot be sent during the handshake",
        )

    def receive_data(self, data: bytes | None) -> None:
        self._h11_connection.receive_data(data or b"")

        while True:
            try:
                event = self._h11_connection.next_event()
            except h11.RemoteProtocolError:
                raise RemoteProtocolError(
                    "Bad HTTP message",
                    event_hint=RejectConnection(),
                )

            if (
                event is h11.NEED_DATA
                or event is h11.PAUSED
                or isinstance(event, h11.ConnectionClosed)
            ):
                return

            if self.client:
                self._receive_client_event(event)
            elif isinstance(event, h11.Request):
                self._events.append(self._process_connection_request(event))

    def events(self) -> Generator[Event, None, None]:
        while self._events:
            yield self._events.popleft()

    def _receive_client_event(self, event: object) -> None:
        if isinstance(event, h11.InformationalResponse):
            if event.status_code == 101:
                self._events.append(self._establish_client_connection(event))
            else:
                self._events.append(
                    RejectConnection(
                        headers=list(event.headers),
                        status_code=event.status_code,
                        has_body=False,
                    ),
                )
                self._state = ConnectionState.CLOSED
        elif isinstance(event, h11.Response):
            self._state = ConnectionState.REJECTING
            self._events.append(
                RejectConnection(
                    headers=list(event.headers),
                    status_code=event.status_code,
                    has_body=True,
                ),
            )
        elif isinstance(event, h11.Data):
            self._events.append(RejectData(data=event.data, body_finished=False))
        elif isinstance(event, h11.EndOfMessage):
            self._events.append(RejectData(data=b"", body_finished=True))
            self._state = ConnectionState.CLOSED

    def _process_connection_request(self, event: h11.Request) -> Request:
        if event.method != b"GET":
            raise RemoteProtocolError(
                "Request method must be GET",
                event_hint=RejectConnection(),
            )

        connection_tokens: list[str] | None = None
        extensions: list[str] = []
        host: str | None = None
        key: bytes | None = None
        subprotocols: list[str] = []
        upgrade = b""
        version: bytes | None = None
        headers: Headers = []

        for name, value in event.headers:
            name = name.lower()
            if name == b"connection":
                connection_tokens = split_comma_header(value)
            elif name == b"host":
                host = value.decode("idna")
                continue
            elif name == b"sec-websocket-extensions":
                extensions.extend(split_comma_header(value))
                continue
            elif name == b"sec-websocket-key":
                key = value
            elif name == b"sec-websocket-protocol":
                subprotocols.extend(split_comma_header(value))
                continue
            elif name == b"sec-websocket-version":
                version = value
            elif name == b"upgrade":
                upgrade = value
            headers.append((name, value))

        if connection_tokens is None or not any(
            token.lower() == "upgrade" for token in connection_tokens
        ):
            raise RemoteProtocolError(
                "Missing header, 'Connection: Upgrade'",
                event_hint=RejectConnection(),
            )

        if version != WEBSOCKET_VERSION:
            raise RemoteProtocolError(
                "Missing header, 'Sec-WebSocket-Version'",
                event_hint=RejectConnection(
                    headers=[(b"Sec-WebSocket-Version", WEBSOCKET_VERSION)],
                    status_code=426 if version else 400,
                ),
            )

        if key is None:
            raise RemoteProtocolError(
                "Missing header, 'Sec-WebSocket-Key'",
                event_hint=RejectConnection(),
            )

        if upgrade.lower() != WEBSOCKET_UPGRADE:
            raise RemoteProtocolError(
                "Missing header, 'Upgrade: websocket'",
                event_hint=RejectConnection(),
            )

        if host is None:
            raise RemoteProtocolError(
                "Missing header, 'Host'",
                event_hint=RejectConnection(),
            )

        self._nonce = key
        request = Request(
            host=host,
            target=event.target,
            extensions=extensions,
            extra_headers=headers,
            subprotocols=subprotocols,
        )
        self._initiating_request = request
        return request

    def _initiate_connection(self, event: Request) -> bytes:
        if not self.client:
            raise LocalProtocolError(
                "Cannot initiate a connection when acting as the server",
            )
        if self._state is not ConnectionState.CONNECTING:
            raise LocalProtocolError("Connection already initiated")

        self._initiating_request = event
        self._nonce = generate_nonce()

        headers: Headers = [
            (b"Host", event.host.encode("idna")),
            (b"Upgrade", WEBSOCKET_UPGRADE),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Key", self._nonce),
            (b"Sec-WebSocket-Version", WEBSOCKET_VERSION),
        ]
        headers.extend(event.extra_headers)

        extension_headers: list[bytes] = []
        for extension in event.extensions:
            value = _extension_header_value(extension, extension.offer())
            if value is not None:
                extension_headers.append(value)
        if extension_headers:
            headers.append(
                (b"Sec-WebSocket-Extensions", b", ".join(extension_headers)),
            )

        if event.subprotocols:
            headers.append(
                (b"Sec-WebSocket-Protocol", ", ".join(event.subprotocols).encode("ascii")),
            )

        request = h11.Request(method=b"GET", target=event.target, headers=headers)
        return cast(bytes, self._h11_connection.send(request))

    def _establish_client_connection(
        self,
        event: h11.InformationalResponse,
    ) -> AcceptConnection:
        if self._initiating_request is None or self._nonce is None:
            raise LocalProtocolError("Connection was not initiated")

        headers = normed_header_dict(event.headers)
        connection_tokens = split_comma_header(headers.get(b"connection", b""))
        if not any(token.lower() == "upgrade" for token in connection_tokens):
            raise RemoteProtocolError(
                "Missing header, 'Connection: Upgrade'",
            )

        if headers.get(b"upgrade", b"").lower() != WEBSOCKET_UPGRADE:
            raise RemoteProtocolError(
                "Missing header, 'Upgrade: websocket'",
            )

        if headers.get(b"sec-websocket-accept") != generate_accept_token(self._nonce):
            raise RemoteProtocolError(
                "Bad accept token",
            )

        subprotocol: str | None = None
        if b"sec-websocket-protocol" in headers:
            subprotocol = headers[b"sec-websocket-protocol"].decode("ascii")
            if subprotocol not in self._initiating_request.subprotocols:
                raise RemoteProtocolError("Unrecognized subprotocol")

        extensions = self._finalize_extensions(
            split_comma_header(headers.get(b"sec-websocket-extensions", b"")),
            self._initiating_request.extensions,
        )

        ignored = {
            b"connection",
            b"upgrade",
            b"sec-websocket-accept",
            b"sec-websocket-protocol",
            b"sec-websocket-extensions",
        }
        extra_headers = [
            (name, value) for name, value in event.headers if name.lower() not in ignored
        ]

        self._connection = Connection(ConnectionType.CLIENT, extensions)
        self._state = ConnectionState.OPEN
        return AcceptConnection(
            subprotocol=subprotocol,
            extensions=extensions,
            extra_headers=extra_headers,
        )

    def _accept(self, event: AcceptConnection) -> bytes:
        if self.client:
            raise LocalProtocolError(
                "Cannot accept a connection when acting as the client",
            )
        if self._initiating_request is None or self._nonce is None:
            raise LocalProtocolError("Connection was not initiated")
        if self._state is not ConnectionState.CONNECTING:
            raise LocalProtocolError("Connection already accepted")

        if (
            event.subprotocol is not None
            and event.subprotocol not in self._initiating_request.subprotocols
        ):
            raise LocalProtocolError("Subprotocol was not requested by the client")

        accepted_extensions, extension_headers = self._accept_extensions(
            event.extensions,
            self._initiating_request.extensions,
        )

        headers: Headers = [
            (b"Upgrade", b"WebSocket"),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Accept", generate_accept_token(self._nonce)),
        ]
        headers.extend(event.extra_headers)
        if event.subprotocol is not None:
            headers.append((b"Sec-WebSocket-Protocol", event.subprotocol.encode("ascii")))
        if extension_headers:
            headers.append((b"Sec-WebSocket-Extensions", b", ".join(extension_headers)))

        response = h11.InformationalResponse(status_code=101, headers=headers)
        data = cast(bytes, self._h11_connection.send(response))
        self._connection = Connection(ConnectionType.SERVER, accepted_extensions)
        self._state = ConnectionState.OPEN
        return data

    def _reject(self, event: RejectConnection) -> bytes:
        if self.client:
            raise LocalProtocolError(
                "Cannot reject a connection when acting as the client",
            )
        if self._state is not ConnectionState.CONNECTING:
            raise LocalProtocolError("Connection is not in a state where it can be rejected")

        response = h11.Response(
            status_code=event.status_code,
            headers=event.headers,
        )
        data = cast(bytes, self._h11_connection.send(response))
        if event.has_body:
            self._state = ConnectionState.REJECTING
        else:
            data += cast(bytes, self._h11_connection.send(h11.EndOfMessage()))
            self._state = ConnectionState.CLOSED
        return data

    def _send_reject_data(self, event: RejectData) -> bytes:
        if self.client:
            raise LocalProtocolError(
                "Cannot send rejection data when acting as the client",
            )
        if self._state is not ConnectionState.REJECTING:
            raise LocalProtocolError("Connection is not rejecting")

        data = cast(bytes, self._h11_connection.send(h11.Data(data=event.data)))
        if event.body_finished:
            data += cast(bytes, self._h11_connection.send(h11.EndOfMessage()))
            self._state = ConnectionState.CLOSED
        return data

    def _accept_extensions(
        self,
        extensions: Sequence[Extension],
        offers: Iterable[str],
    ) -> tuple[list[Extension], list[bytes]]:
        offered = list(offers)
        accepted: list[Extension] = []
        headers: list[bytes] = []

        for extension in extensions:
            for offer in offered:
                name, parameters = _extension_parts(offer)
                if name != extension.name:
                    continue

                value = extension.accept(parameters)
                header = _extension_header_value(extension, value)
                if header is not None:
                    final_parameters = ""
                    if value is not True and value != "":
                        final_parameters = str(value)
                    extension.finalize(final_parameters)
                    accepted.append(extension)
                    headers.append(header)
                break

        return accepted, headers

    def _finalize_extensions(
        self,
        values: Iterable[str],
        offered_extensions: Sequence[Extension],
    ) -> list[Extension]:
        accepted: list[Extension] = []
        used_names: set[str] = set()

        for value in values:
            name, parameters = _extension_parts(value)
            if name in used_names:
                raise RemoteProtocolError("Duplicate extension in response")

            extension = next(
                (
                    candidate
                    for candidate in offered_extensions
                    if candidate.name == name
                ),
                None,
            )
            if extension is None:
                raise RemoteProtocolError("Unrecognized extension in response")

            try:
                extension.finalize(parameters)
            except Exception as exc:
                raise RemoteProtocolError("Invalid extension in response") from exc

            used_names.add(name)
            accepted.append(extension)

        return accepted