import abc
from collections.abc import Callable

__all__ = ["Thenable"]


class Thenable(Callable, metaclass=abc.ABCMeta):
    __slots__ = ()

    @abc.abstractmethod
    def then(self, on_success, on_error=None):
        raise NotImplementedError

    @abc.abstractmethod
    def throw(self, exc=None, tb=None, propagate=True):
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, candidate):
        if cls is Thenable:
            if any("then" in base.__dict__ for base in candidate.__mro__):
                return True
        return NotImplemented

    @classmethod
    def register(cls, candidate):
        type(cls).register(cls, candidate)
        return candidate


@Thenable.register
class ThenableProxy:
    def _set_promise_target(self, promise):
        self._p = promise

    def then(self, on_success, on_error=None):
        return self._p.then(on_success, on_error)

    def cancel(self):
        return self._p.cancel()

    def throw1(self, exc=None):
        return self._p.throw1(exc)

    def throw(self, exc=None, tb=None, propagate=True):
        return self._p.throw(exc, tb=tb, propagate=propagate)

    @property
    def cancelled(self):
        return self._p.cancelled

    @property
    def ready(self):
        return self._p.ready

    @property
    def failed(self):
        return self._p.failed