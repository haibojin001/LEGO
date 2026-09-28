from typing import Dict, Optional, Set, Tuple, Type, Union, cast

from ._events import *
from ._util import LocalProtocolError, Sentinel

__all__ = [
    "CLIENT",
    "SERVER",
    "IDLE",
    "SEND_RESPONSE",
    "SEND_BODY",
    "DONE",
    "MUST_CLOSE",
    "CLOSED",
    "MIGHT_SWITCH_PROTOCOL",
    "SWITCHED_PROTOCOL",
    "ERROR",
]


class CLIENT(Sentinel, metaclass=Sentinel):
    pass


class SERVER(Sentinel, metaclass=Sentinel):
    pass


class IDLE(Sentinel, metaclass=Sentinel):
    pass


class SEND_RESPONSE(Sentinel, metaclass=Sentinel):
    pass


class SEND_BODY(Sentinel, metaclass=Sentinel):
    pass


class DONE(Sentinel, metaclass=Sentinel):
    pass


class MUST_CLOSE(Sentinel, metaclass=Sentinel):
    pass


class CLOSED(Sentinel, metaclass=Sentinel):
    pass


class ERROR(Sentinel, metaclass=Sentinel):
    pass


class MIGHT_SWITCH_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass


class SWITCHED_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass


class _SWITCH_UPGRADE(Sentinel, metaclass=Sentinel):
    pass


class _SWITCH_CONNECT(Sentinel, metaclass=Sentinel):
    pass


EventTransitionType = Dict[
    Type[Sentinel],
    Dict[
        Type[Sentinel],
        Dict[
            Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]],
            Type[Sentinel],
        ],
    ],
]

EVENT_TRIGGERED_TRANSITIONS: EventTransitionType = {
    CLIENT: {
        IDLE: {
            Request: SEND_BODY,
            ConnectionClosed: CLOSED,
        },
        SEND_BODY: {
            Data: SEND_BODY,
            EndOfMessage: DONE,
        },
        DONE: {
            ConnectionClosed: CLOSED,
        },
        MUST_CLOSE: {
            ConnectionClosed: CLOSED,
        },
        CLOSED: {
            ConnectionClosed: CLOSED,
        },
        MIGHT_SWITCH_PROTOCOL: {},
        SWITCHED_PROTOCOL: {},
        ERROR: {},
    },
    SERVER: {
        IDLE: {
            ConnectionClosed: CLOSED,
            Response: SEND_BODY,
            (Request, CLIENT): SEND_RESPONSE,
        },
        SEND_RESPONSE: {
            InformationalResponse: SEND_RESPONSE,
            Response: SEND_BODY,
            (InformationalResponse, _SWITCH_UPGRADE): SWITCHED_PROTOCOL,
            (Response, _SWITCH_CONNECT): SWITCHED_PROTOCOL,
        },
        SEND_BODY: {
            Data: SEND_BODY,
            EndOfMessage: DONE,
        },
        DONE: {
            ConnectionClosed: CLOSED,
        },
        MUST_CLOSE: {
            ConnectionClosed: CLOSED,
        },
        CLOSED: {
            ConnectionClosed: CLOSED,
        },
        SWITCHED_PROTOCOL: {},
        ERROR: {},
    },
}

StateTransitionType = Dict[
    Tuple[Type[Sentinel], Type[Sentinel]],
    Dict[Type[Sentinel], Type[Sentinel]],
]

STATE_TRIGGERED_TRANSITIONS: StateTransitionType = {
    (MIGHT_SWITCH_PROTOCOL, SWITCHED_PROTOCOL): {
        CLIENT: SWITCHED_PROTOCOL,
    },
    (CLOSED, DONE): {
        SERVER: MUST_CLOSE,
    },
    (CLOSED, IDLE): {
        SERVER: MUST_CLOSE,
    },
    (ERROR, DONE): {
        SERVER: MUST_CLOSE,
    },
    (DONE, CLOSED): {
        CLIENT: MUST_CLOSE,
    },
    (IDLE, CLOSED): {
        CLIENT: MUST_CLOSE,
    },
    (DONE, ERROR): {
        CLIENT: MUST_CLOSE,
    },
}


class ConnectionState:
    def __init__(self) -> None:
        self.keep_alive = True
        self.pending_switch_proposals: Set[Type[Sentinel]] = set()
        self.states: Dict[Type[Sentinel], Type[Sentinel]] = {
            CLIENT: IDLE,
            SERVER: IDLE,
        }

    def process_error(self, role: Type[Sentinel]) -> None:
        self.states[role] = ERROR
        self._fire_state_triggered_transitions()

    def process_keep_alive_disabled(self) -> None:
        self.keep_alive = False
        self._fire_state_triggered_transitions()

    def process_client_switch_proposal(
        self, switch_event: Type[Sentinel]
    ) -> None:
        self.pending_switch_proposals.add(switch_event)
        self._fire_state_triggered_transitions()

    def process_event(
        self,
        role: Type[Sentinel],
        event_type: Type[Event],
        server_switch_event: Optional[Type[Sentinel]] = None,
    ) -> None:
        transition_event: Union[
            Type[Event], Tuple[Type[Event], Type[Sentinel]]
        ] = event_type

        if server_switch_event is not None:
            transition_event = cast(
                Tuple[Type[Event], Type[Sentinel]],
                (event_type, server_switch_event),
            )

        self._fire_event_triggered_transition(role, transition_event)

        if role is CLIENT and event_type is Request:
            self._fire_event_triggered_transition(SERVER, (Request, CLIENT))

        if role is SERVER and event_type is Response and server_switch_event is None:
            self.pending_switch_proposals.clear()

        self._fire_state_triggered_transitions()

    def _fire_event_triggered_transition(
        self,
        role: Type[Sentinel],
        event_type: Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]],
    ) -> None:
        old_state = self.states[role]

        try:
            next_state = EVENT_TRIGGERED_TRANSITIONS[role][old_state][event_type]
        except KeyError:
            event_class: Type[Event]
            if isinstance(event_type, tuple):
                event_class = event_type[0]
            else:
                event_class = event_type
            raise LocalProtocolError(
                "can't handle event type {} when role={} and state={}".format(
                    event_class.__name__, role, old_state
                )
            )

        self.states[role] = next_state

    def _fire_state_triggered_transitions(self) -> None:
        while True:
            if (
                self.pending_switch_proposals
                and self.states[CLIENT] is DONE
            ):
                self.states[CLIENT] = MIGHT_SWITCH_PROTOCOL
                continue

            paired_state = (self.states[CLIENT], self.states[SERVER])
            transitions = STATE_TRIGGERED_TRANSITIONS.get(paired_state)
            if transitions is not None:
                for role, new_state in transitions.items():
                    self.states[role] = new_state
                continue

            if not self.keep_alive:
                changed = False
                for role in (CLIENT, SERVER):
                    if self.states[role] is DONE:
                        self.states[role] = MUST_CLOSE
                        changed = True
                if changed:
                    continue

            break