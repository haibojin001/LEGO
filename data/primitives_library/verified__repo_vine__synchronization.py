"""Synchronization primitives."""

from .abstract import Thenable
from .promises import promise

__all__ = ["barrier"]


class barrier:
    """Collect completion notifications from a group of promises."""

    __slots__ = (
        "p",
        "args",
        "kwargs",
        "_value",
        "size",
        "ready",
        "reason",
        "cancelled",
        "finalized",
        "__weakref__",
        "__dict__",
    )

    def __init__(self, promises=None, args=None, kwargs=None,
                 callback=None, size=None):
        self.p = promise()
        self.args = args or ()
        self.kwargs = kwargs or {}
        self._value = 0
        self.size = size or 0

        if not self.size and promises:
            count = promises.__len__()
            if count is not NotImplemented:
                self.size = count

        self.ready = False
        self.failed = False
        self.reason = None
        self.cancelled = False
        self.finalized = False

        for item in promises or ():
            self.add_noincr(item)

        self.finalized = bool(promises or self.size)

        if callback:
            self.then(callback)

    def __call__(self, *args, **kwargs):
        if self.ready or self.cancelled:
            return

        self._value += 1
        if self.finalized and self._value >= self.size:
            self.ready = True
            self.p(*self.args, **self.kwargs)

    def finalize(self):
        if not self.finalized and self._value >= self.size:
            self.p(*self.args, **self.kwargs)
        self.finalized = True

    def cancel(self):
        self.cancelled = True
        self.p.cancel()

    def add_noincr(self, p):
        if self.cancelled:
            return
        if self.ready:
            raise ValueError("Cannot add promise to full barrier")
        p.then(self)

    def add(self, p):
        if self.cancelled:
            return
        self.add_noincr(p)
        self.size += 1

    def then(self, callback, errback=None):
        self.p.then(callback, errback)

    def throw(self, *args, **kwargs):
        if not self.cancelled:
            self.p.throw(*args, **kwargs)

    throw1 = throw


Thenable.register(barrier)