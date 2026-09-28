from __future__ import annotations

import collections.abc as c
import sys
import typing as t
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from inspect import iscoroutinefunction

from ._utilities import Symbol
from ._utilities import make_id
from ._utilities import make_ref

F = t.TypeVar("F", bound=c.Callable[..., t.Any])

ANY = Symbol("ANY")
ANY_ID = 0


class Signal:
    """A notification emitter."""

    ANY = ANY
    set_class: type[set[t.Any]] = set

    @cached_property
    def receiver_connected(self) -> Signal:
        return Signal(doc="Emitted after a receiver connects.")

    @cached_property
    def receiver_disconnected(self) -> Signal:
        return Signal(doc="Emitted after a receiver disconnects.")

    def __init__(self, doc: str | None = None) -> None:
        if doc:
            self.__doc__ = doc

        self.receivers: dict[
            t.Any, weakref.ref[c.Callable[..., t.Any]] | c.Callable[..., t.Any]
        ] = {}
        self.is_muted = False
        self._by_receiver: dict[t.Any, set[t.Any]] = defaultdict(self.set_class)
        self._by_sender: dict[t.Any, set[t.Any]] = defaultdict(self.set_class)
        self._weak_senders: dict[t.Any, weakref.ref[t.Any]] = {}

    def connect(self, receiver: F, sender: t.Any = ANY, weak: bool = True) -> F:
        receiver_id = make_id(receiver)
        sender_id = ANY_ID if sender is ANY else make_id(sender)

        if weak:
            self.receivers[receiver_id] = make_ref(
                receiver, self._make_cleanup_receiver(receiver_id)
            )
        else:
            self.receivers[receiver_id] = receiver

        self._by_sender[sender_id].add(receiver_id)
        self._by_receiver[receiver_id].add(sender_id)

        if sender is not ANY and sender_id not in self._weak_senders:
            try:
                self._weak_senders[sender_id] = make_ref(
                    sender, self._make_cleanup_sender(sender_id)
                )
            except TypeError:
                pass

        if "receiver_connected" in self.__dict__ and self.receiver_connected.receivers:
            try:
                self.receiver_connected.send(
                    self, receiver=receiver, sender=sender, weak=weak
                )
            except TypeError:
                self.disconnect(receiver, sender)
                raise

        return receiver

    def connect_via(self, sender: t.Any, weak: bool = False) -> c.Callable[[F], F]:
        def decorator(fn: F) -> F:
            self.connect(fn, sender, weak)
            return fn

        return decorator

    @contextmanager
    def connected_to(
        self, receiver: c.Callable[..., t.Any], sender: t.Any = ANY
    ) -> c.Generator[None, None, None]:
        self.connect(receiver, sender=sender, weak=False)

        try:
            yield None
        finally:
            self.disconnect(receiver)

    @contextmanager
    def muted(self) -> c.Generator[None, None, None]:
        self.is_muted = True

        try:
            yield None
        finally:
            self.is_muted = False

    def send(
        self,
        sender: t.Any | None = None,
        /,
        *,
        _async_wrapper: c.Callable[
            [c.Callable[..., c.Coroutine[t.Any, t.Any, t.Any]]],
            c.Callable[..., t.Any],
        ]
        | None = None,
        **kwargs: t.Any,
    ) -> list[tuple[c.Callable[..., t.Any], t.Any]]:
        if self.is_muted:
            return []

        results: list[tuple[c.Callable[..., t.Any], t.Any]] = []

        for receiver in self.receivers_for(sender):
            if iscoroutinefunction(receiver):
                if _async_wrapper is None:
                    raise RuntimeError("Cannot send to a coroutine function.")

                receiver = _async_wrapper(receiver)

            results.append((receiver, receiver(sender, **kwargs)))

        return results

    async def send_async(
        self,
        sender: t.Any | None = None,
        /,
        *,
        _sync_wrapper: c.Callable[
            [c.Callable[..., t.Any]],
            c.Callable[..., c.Coroutine[t.Any, t.Any, t.Any]],
        ]
        | None = None,
        **kwargs: t.Any,
    ) -> list[tuple[c.Callable[..., t.Any], t.Any]]:
        if self.is_muted:
            return []

        results: list[tuple[c.Callable[..., t.Any], t.Any]] = []

        for receiver in self.receivers_for(sender):
            if iscoroutinefunction(receiver):
                results.append((receiver, await receiver(sender, **kwargs)))
            else:
                if _sync_wrapper is None:
                    raise RuntimeError("Cannot send to a non-coroutine function.")

                receiver = _sync_wrapper(receiver)
                results.append((receiver, await receiver(sender, **kwargs)))

        return results

    def has_receivers_for(self, sender: t.Any) -> bool:
        if sender is ANY:
            return bool(self._by_sender[ANY_ID])

        sender_id = make_id(sender)
        return bool(self._by_sender[ANY_ID] | self._by_sender[sender_id])

    def receivers_for(
        self, sender: t.Any
    ) -> c.Generator[c.Callable[..., t.Any], None, None]:
        if not self.receivers:
            return

        sender_id = make_id(sender)

        if sender_id in self._by_sender:
            receiver_ids = self._by_sender[ANY_ID] | self._by_sender[sender_id]
        else:
            receiver_ids = self._by_sender[ANY_ID].copy()

        for receiver_id in receiver_ids:
            receiver = self.receivers.get(receiver_id)

            if receiver is None:
                continue

            if isinstance(receiver, weakref.ref):
                strong_receiver = receiver()

                if strong_receiver is None:
                    self._disconnect_all(receiver_id)
                    continue

                receiver = strong_receiver

            yield receiver

    def disconnect(self, receiver: c.Callable[..., t.Any], sender: t.Any = ANY) -> None:
        receiver_id = make_id(receiver)

        if sender is ANY:
            for sender_id in self._by_receiver[receiver_id].copy():
                self._disconnect(receiver_id, sender_id)
        else:
            self._disconnect(receiver_id, make_id(sender))

        if (
            "receiver_disconnected" in self.__dict__
            and self.receiver_disconnected.receivers
        ):
            self.receiver_disconnected.send(
                self, receiver=receiver, sender=sender
            )

    def _disconnect(self, receiver_id: t.Any, sender_id: t.Any) -> None:
        self._by_sender[sender_id].discard(receiver_id)
        self._by_receiver[receiver_id].discard(sender_id)

        if not self._by_sender[sender_id]:
            self._by_sender.pop(sender_id, None)
            self._weak_senders.pop(sender_id, None)

        if not self._by_receiver[receiver_id]:
            self._by_receiver.pop(receiver_id, None)
            self.receivers.pop(receiver_id, None)

    def _disconnect_all(self, receiver_id: t.Any) -> None:
        for sender_id in self._by_receiver.get(receiver_id, ()).copy():
            self._disconnect(receiver_id, sender_id)

    def _make_cleanup_receiver(
        self, receiver_id: t.Any
    ) -> c.Callable[[weakref.ref[t.Any]], None]:
        self_ref = weakref.ref(self)

        def cleanup(ref: weakref.ref[t.Any]) -> None:
            if sys.is_finalizing():
                return

            signal = self_ref()

            if signal is not None:
                signal._disconnect_all(receiver_id)

        return cleanup

    def _make_cleanup_sender(
        self, sender_id: t.Any
    ) -> c.Callable[[weakref.ref[t.Any]], None]:
        self_ref = weakref.ref(self)

        def cleanup(ref: weakref.ref[t.Any]) -> None:
            if sys.is_finalizing():
                return

            signal = self_ref()

            if signal is None:
                return

            signal._weak_senders.pop(sender_id, None)

            for receiver_id in signal._by_sender.get(sender_id, ()).copy():
                signal._disconnect(receiver_id, sender_id)

        return cleanup

    def _cleanup_bookkeeping(self) -> None:
        for mapping in (self._by_sender, self._by_receiver):
            for key, values in list(mapping.items()):
                if not values:
                    mapping.pop(key, None)


class NamedSignal(Signal):
    """A named notification emitter."""

    def __init__(self, name: str, doc: str | None = None) -> None:
        super().__init__(doc)
        self.name = name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}; {len(self.receivers)} receivers>"


class Namespace(dict[str, NamedSignal]):
    """A mapping of named signals."""

    def signal(self, name: str, doc: str | None = None) -> NamedSignal:
        if name not in self:
            self[name] = NamedSignal(name, doc)

        return self[name]


class WeakNamespace(weakref.WeakValueDictionary[str, NamedSignal]):
    """A weak mapping of named signals."""

    def signal(self, name: str, doc: str | None = None) -> NamedSignal:
        signal = self.get(name)

        if signal is None:
            signal = NamedSignal(name, doc)
            self[name] = signal

        return signal


__all__ = [
    "ANY",
    "ANY_ID",
    "NamedSignal",
    "Namespace",
    "Signal",
    "WeakNamespace",
]