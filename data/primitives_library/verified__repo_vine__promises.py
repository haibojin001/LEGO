import inspect
import sys
from collections import deque
from weakref import WeakMethod, ref

from .abstract import Thenable
from .utils import reraise

__all__ = ["promise"]


@Thenable.register
class promise:
    if not hasattr(sys, "pypy_version_info"):
        __slots__ = (
            "fun",
            "args",
            "kwargs",
            "ready",
            "failed",
            "value",
            "ignore_result",
            "reason",
            "_svpending",
            "_lvpending",
            "on_error",
            "cancelled",
            "weak",
            "__weakref__",
            "__dict__",
        )

    def __init__(
        self,
        fun=None,
        args=None,
        kwargs=None,
        callback=None,
        on_error=None,
        weak=False,
        ignore_result=False,
    ):
        self.weak = weak
        self.ignore_result = ignore_result
        self.fun = self._get_fun_or_weakref(fun, weak)
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.ready = False
        self.failed = False
        self.value = None
        self.reason = None
        self._svpending = None
        self._lvpending = None
        self.on_error = on_error
        self.cancelled = False

        if callback is not None:
            self.then(callback)

        if self.fun:
            assert callable(fun)

    @staticmethod
    def _get_fun_or_weakref(fun, weak):
        if not weak:
            return fun
        if inspect.ismethod(fun):
            return WeakMethod(fun)
        return ref(fun)

    def __repr__(self):
        identity = f"{type(self).__name__}@0x{id(self):x}"
        if self.fun:
            return f"<{identity} --> {self.fun!r}>"
        return f"<{identity}>"

    def _fun_is_alive(self, fun):
        if self.weak:
            return fun()
        return self.fun

    def cancel(self):
        self.cancelled = True
        try:
            one = self._svpending
            if one is not None:
                one.cancel()

            many = self._lvpending
            if many is not None:
                for item in many:
                    item.cancel()

            if isinstance(self.on_error, Thenable):
                self.on_error.cancel()
        finally:
            self._svpending = None
            self._lvpending = None
            self.on_error = None

    def __call__(self, *args, **kwargs):
        result = None

        if self.cancelled:
            return None

        call_args = self.args + args if args else self.args
        call_kwargs = dict(self.kwargs, **kwargs) if kwargs else self.kwargs
        callable_ = self._fun_is_alive(self.fun)

        if callable_ is not None:
            try:
                if self.ignore_result:
                    callable_(*call_args, **call_kwargs)
                    pending_args, pending_kwargs = (), {}
                else:
                    result = callable_(*call_args, **call_kwargs)
                    pending_args, pending_kwargs = (result,), {}
                    self.value = (pending_args, pending_kwargs)
            except Exception:
                return self.throw()
        else:
            pending_args, pending_kwargs = call_args, call_kwargs
            self.value = (pending_args, pending_kwargs)

        self.ready = True

        single = self._svpending
        if single is not None:
            try:
                single(*pending_args, **pending_kwargs)
            finally:
                self._svpending = None
        else:
            multiple = self._lvpending
            try:
                while multiple:
                    multiple.popleft()(*pending_args, **pending_kwargs)
            finally:
                self._lvpending = None

        return result

    def then(self, callback, on_error=None):
        if not isinstance(callback, Thenable):
            callback = promise(callback, on_error=on_error)

        if self.cancelled:
            callback.cancel()
            return callback

        if self.failed:
            callback.throw(self.reason)
        elif self.ready:
            saved_args, saved_kwargs = self.value
            callback(*saved_args, **saved_kwargs)

        if self._lvpending is None:
            prior = self._svpending
            if prior is not None:
                self._svpending = None
                self._lvpending = deque([prior])
            else:
                self._svpending = callback
                return callback

        self._lvpending.append(callback)
        return callback

    def throw1(self, exc=None):
        if self.cancelled:
            return

        if exc is None:
            exc = sys.exc_info()[1]

        self.failed = True
        self.reason = exc

        if self.on_error:
            self.on_error(*self.args + (exc,), **self.kwargs)

    def throw(self, exc=None, tb=None, propagate=True):
        if self.cancelled:
            return

        active_exception = sys.exc_info()[1]
        if exc is None:
            exc = active_exception

        try:
            self.throw1(exc)

            single = self._svpending
            if single is not None:
                try:
                    single.throw1(exc)
                finally:
                    self._svpending = None
            else:
                multiple = self._lvpending
                try:
                    while multiple:
                        multiple.popleft().throw1(exc)
                finally:
                    self._lvpending = None
        finally:
            if self.on_error is None and propagate:
                if tb is None and (exc is None or exc is active_exception):
                    raise
                reraise(type(exc), exc, tb)

    @property
    def listeners(self):
        if self._lvpending:
            return self._lvpending
        return [self._svpending]