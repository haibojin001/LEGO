from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from ._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
    _SWITCH_CONNECT,
    _SWITCH_UPGRADE,
    CLIENT,
    ConnectionState,
    DONE,
    ERROR,
    MIGHT_SWITCH_PROTOCOL,
    SEND_BODY,
    SERVER,
    SWITCHED_PROTOCOL,
)
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel
from ._writers import WRITERS, WritersType

__all__ = ["Connection", "NEED_DATA", "PAUSED"]


class NEED_DATA(Sentinel, metaclass=Sentinel):
    pass


class PAUSED(Sentinel, metaclass=Sentinel):
    pass


DEFAULT_MAX_INCOMPLETE_EVENT_SIZE = 16 * 1024


def _keep_alive(event: Union[Request, Response]) -> bool:
    values = get_comma_header(event.headers, b"connection")
    if b"close" in values:
        return False
    return getattr(event, "http_version", b"1.1") >= b"1.1"


def _body_framing(
    request_method: bytes, event: Union[Request, Response]
) -> Tuple[str, Union[Tuple[()], Tuple[int]]]:
    assert type(event) in (Request, Response)

    if type(event) is Response:
        if (
            event.status_code in (204, 304)
            or request_method == b"HEAD"
            or (
                request_method == b"CONNECT"
                and 200 <= event.status_code < 300
            )
        ):
            return ("content-length", (0,))
        assert event.status_code >= 200

    transfer_encoding = get_comma_header(event.headers, b"transfer-encoding")
    if transfer_encoding:
        assert transfer_encoding == [b"chunked"]
        return ("chunked", ())

    content_length = get_comma_header(event.headers, b"content-length")
    if content_length:
        return ("content-length", (int(content_length[0]),))

    if type(event) is Request:
        return ("content-length", (0,))
    return ("http/1.0", ())


class Connection:
    """Stateful HTTP/1.x protocol engine."""

    def __init__(
        self,
        our_role: Type[Sentinel],
        max_incomplete_event_size: int = DEFAULT_MAX_INCOMPLETE_EVENT_SIZE,
    ) -> None:
        if our_role not in (CLIENT, SERVER):
            raise ValueError(f"expected CLIENT or SERVER, not {our_role!r}")

        self._max_incomplete_event_size = max_incomplete_event_size
        self.our_role = our_role
        self.their_role: Type[Sentinel] = SERVER if our_role is CLIENT else CLIENT

        self._cstate = ConnectionState()
        self._writer = self._get_io_object(self.our_role, None, WRITERS)
        self._reader = self._get_io_object(self.their_role, None, READERS)

        self._receive_buffer = ReceiveBuffer()
        self._receive_buffer_closed = False

        self.their_http_version: Optional[bytes] = None
        self._request_method: Optional[bytes] = None
        self.client_is_waiting_for_100_continue = False

    @property
    def states(self) -> Dict[Type[Sentinel], Type[Sentinel]]:
        return dict(self._cstate.states)

    @property
    def our_state(self) -> Type[Sentinel]:
        return self._cstate.states[self.our_role]

    @property
    def their_state(self) -> Type[Sentinel]:
        return self._cstate.states[self.their_role]

    @property
    def they_are_waiting_for_100_continue(self) -> bool:
        return (
            self.their_role is CLIENT
            and self.client_is_waiting_for_100_continue
        )

    @property
    def trailing_data(self) -> Tuple[bytes, bool]:
        return (bytes(self._receive_buffer), self._receive_buffer_closed)

    def start_next_cycle(self) -> None:
        old_states = dict(self._cstate.states)
        self._cstate.start_next_cycle()
        self._request_method = None
        assert not self.client_is_waiting_for_100_continue
        self._respond_to_state_changes(old_states)

    def _process_error(self, role: Type[Sentinel]) -> None:
        old_states = dict(self._cstate.states)
        self._cstate.process_error(role)
        self._respond_to_state_changes(old_states)

    def _server_switch_event(self, event: Event) -> Optional[Type[Sentinel]]:
        if type(event) is InformationalResponse and event.status_code == 101:
            return _SWITCH_UPGRADE

        if type(event) is Response:
            if (
                _SWITCH_CONNECT in self._cstate.pending_switch_proposals
                and 200 <= event.status_code < 300
            ):
                return _SWITCH_CONNECT

        return None

    def _request_switch_event(self, event: Request) -> Optional[Type[Sentinel]]:
        if event.method == b"CONNECT":
            return _SWITCH_CONNECT
        if b"upgrade" in get_comma_header(event.headers, b"connection"):
            return _SWITCH_UPGRADE
        return None

    def _get_io_object(
        self,
        role: Type[Sentinel],
        event: Optional[Event],
        io_dict: Union[ReadersType, WritersType],
    ) -> Any:
        state = self._cstate.states[role]
        role_io = io_dict[role]

        if state is SEND_BODY:
            assert event is not None
            assert self._request_method is not None
            framing, args = _body_framing(
                self._request_method,
                cast(Union[Request, Response], event),
            )
            return role_io[state][framing](*args)

        return role_io[state]

    def _respond_to_state_changes(
        self,
        old_states: Dict[Type[Sentinel], Type[Sentinel]],
        event: Optional[Event] = None,
    ) -> None:
        if old_states[self.our_role] is not self.our_state:
            self._writer = self._get_io_object(self.our_role, event, WRITERS)

        if old_states[self.their_role] is not self.their_state:
            self._reader = self._get_io_object(self.their_role, event, READERS)

    def _process_event(self, role: Type[Sentinel], event: Event) -> None:
        old_states = dict(self._cstate.states)

        if type(event) is Request:
            self._request_method = event.method
            if role is self.their_role:
                self.their_http_version = event.http_version
            if has_expect_100_continue(event.headers):
                self.client_is_waiting_for_100_continue = True

            switch = self._request_switch_event(event)
            if switch is not None:
                self._cstate.process_switch_proposal(role, switch)

        elif type(event) is Response:
            if role is self.their_role:
                self.their_http_version = event.http_version

            switch = self._server_switch_event(event)
            if switch is not None:
                self._cstate.process_switch_proposal(role, switch)

        elif type(event) is InformationalResponse:
            if event.status_code == 100:
                self.client_is_waiting_for_100_continue = False

            switch = self._server_switch_event(event)
            if switch is not None:
                self._cstate.process_switch_proposal(role, switch)

        if type(event) in (Request, Response):
            self._cstate.process_keep_alive(role, _keep_alive(event))

        self._cstate.process_event(role, type(event))
        self._respond_to_state_changes(old_states, event)

    def receive_data(self, data: Optional[bytes]) -> None:
        if data is None:
            self._receive_buffer_closed = True
            return

        if self._receive_buffer_closed:
            raise RuntimeError("received data after receiving EOF")

        self._receive_buffer += data

    def _extract_next_receive_event(
        self,
    ) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
        if self._reader is None:
            return PAUSED

        try:
            event = self._reader(self._receive_buffer)
        except LocalProtocolError as exc:
            raise RemoteProtocolError(str(exc)) from exc

        if event is not None:
            return event

        if not self._receive_buffer_closed:
            if len(self._receive_buffer) > self._max_incomplete_event_size:
                raise RemoteProtocolError(
                    "Receive buffer too long",
                    error_status_hint=431,
                )
            return NEED_DATA

        if self._receive_buffer:
            raise RemoteProtocolError("peer unexpectedly closed connection")

        read_eof = getattr(self._reader, "read_eof", None)
        if read_eof is not None:
            try:
                event = read_eof()
            except LocalProtocolError as exc:
                raise RemoteProtocolError(str(exc)) from exc
            if event is not None:
                return event

        if self.their_state is DONE:
            return ConnectionClosed()

        raise RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

    def next_event(self) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
        if self.their_state is DONE:
            return PAUSED

        if self.their_state in (MIGHT_SWITCH_PROTOCOL, SWITCHED_PROTOCOL):
            return PAUSED

        try:
            event = self._extract_next_receive_event()
        except RemoteProtocolError:
            self._process_error(self.their_role)
            raise

        if event in (NEED_DATA, PAUSED):
            return event

        if type(event) is ConnectionClosed:
            return event

        try:
            self._process_event(self.their_role, cast(Event, event))
        except LocalProtocolError as exc:
            self._process_error(self.their_role)
            raise RemoteProtocolError(str(exc)) from exc

        return cast(Event, event)

    def send(self, event: Event) -> bytes:
        return b"".join(self.send_with_data_passthrough(event))

    def send_with_data_passthrough(self, event: Event) -> List[bytes]:
        if self.our_state is ERROR:
            raise LocalProtocolError("Can't send data when our state is ERROR")

        if self._writer is None:
            raise LocalProtocolError(
                "Can't send data when our state is {}.".format(self.our_state)
            )

        writer = self._writer
        data_list: List[bytes] = []

        try:
            self._process_event(self.our_role, event)
            writer(event, data_list)
        except LocalProtocolError:
            self._process_error(self.our_role)
            raise

        return data_list