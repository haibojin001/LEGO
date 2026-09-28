"""Functional helpers for promises."""

from .abstract import Thenable
from .promises import promise


__all__ = [
    "maybe_promise",
    "ensure_promise",
    "ppartial",
    "preplace",
    "ready_promise",
    "starpromise",
    "transform",
    "wrap",
]


def maybe_promise(p):
    if not p:
        return p
    if isinstance(p, Thenable):
        return p
    return promise(p)


def ensure_promise(p):
    if p is None:
        return promise()
    return maybe_promise(p)


def ppartial(p, *args, **kwargs):
    result = ensure_promise(p)
    if args:
        result.args = args + result.args
    if kwargs:
        result.kwargs.update(kwargs)
    return result


def preplace(p, *args, **kwargs):
    def replace_arguments(*_ignored_args, **_ignored_kwargs):
        return p(*args, **kwargs)

    return promise(replace_arguments)


def ready_promise(callback=None, *args):
    result = ensure_promise(callback)
    result(*args)
    return result


def starpromise(fun, *args, **kwargs):
    return promise(fun, args, kwargs)


def transform(filter_, callback, *filter_args, **filter_kwargs):
    callback_promise = ensure_promise(callback)
    result = promise(
        _transback,
        (filter_, callback_promise, filter_args, filter_kwargs),
    )
    result.then(promise(), callback_promise.throw)
    return result


def _transback(filter_, callback, args, kwargs, ret):
    try:
        value = filter_(*(args + (ret,)), **kwargs)
    except Exception:
        callback.throw()
    else:
        return callback(value)


def wrap(p):
    def wrapped(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], promise):
            return args[0].then(p)
        return p(*args, **kwargs)

    return wrapped