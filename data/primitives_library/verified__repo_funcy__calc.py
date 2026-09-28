from datetime import timedelta
import time
import inspect
from collections import deque
from bisect import bisect

from .decorators import wraps


__all__ = ['memoize', 'make_lookuper', 'silent_lookuper', 'cache']


class SkipMemory(Exception):
    pass


def memoize(_func=None, *, key_func=None):
    """Decorate a function so that results are stored and reused."""
    if _func is not None:
        return memoize()(_func)
    return _memory_decorator({}, key_func)


memoize.skip = SkipMemory


def cache(timeout, *, key_func=None):
    """Decorate a function so that results are cached for timeout seconds."""
    if isinstance(timeout, timedelta):
        timeout = timeout.total_seconds()
    return _memory_decorator(CacheMemory(timeout), key_func)


cache.skip = SkipMemory


def _memory_decorator(memory, key_func):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func is not None:
                key = key_func(*args, **kwargs)
            elif kwargs:
                key = args + tuple(sorted(kwargs.items()))
            else:
                key = args

            try:
                return memory[key]
            except KeyError:
                try:
                    result = func(*args, **kwargs)
                    memory[key] = result
                    return result
                except SkipMemory as exc:
                    return exc.args[0] if exc.args else None

        def invalidate(*args, **kwargs):
            if key_func is not None:
                key = key_func(*args, **kwargs)
            elif kwargs:
                key = args + tuple(sorted(kwargs.items()))
            else:
                key = args
            memory.pop(key, None)

        def invalidate_all():
            memory.clear()

        wrapper.invalidate = invalidate
        wrapper.invalidate_all = invalidate_all
        wrapper.memory = memory
        return wrapper

    return decorator


class CacheMemory(dict):
    def __init__(self, timeout):
        self.timeout = timeout
        self.clear()

    def __setitem__(self, key, value):
        expiry = time.time() + self.timeout
        dict.__setitem__(self, key, (value, expiry))
        self._keys.append(key)
        self._expires.append(expiry)

    def __getitem__(self, key):
        value, expiry = dict.__getitem__(self, key)
        if expiry <= time.time():
            self.expire()
            raise KeyError(key)
        return value

    def expire(self):
        count = bisect(self._expires, time.time())
        for _ in range(count):
            self._expires.popleft()
            self.pop(self._keys.popleft(), None)

    def clear(self):
        dict.clear(self)
        self._keys = deque()
        self._expires = deque()


def has_arg_types(func):
    parameters = inspect.signature(func).parameters.values()
    has_args = any(
        parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.VAR_POSITIONAL,
        )
        for parameter in parameters
    )
    has_keys = any(
        parameter.kind in (
            parameter.KEYWORD_ONLY,
            parameter.VAR_KEYWORD,
        )
        for parameter in inspect.signature(func).parameters.values()
    )
    return has_args, has_keys


def _make_lookuper(silent):
    def make_lookuper(func):
        """
        Create a one-argument lookup function backed by a lazily built table.
        """
        has_args, has_keys = has_arg_types(func)
        assert not has_keys, (
            'Lookup table building function should not have keyword arguments'
        )

        if has_args:
            @memoize
            def wrapper(*args):
                builder = lambda: func(*args)
                builder.__name__ = '%s(%s)' % (
                    func.__name__,
                    ', '.join(map(str, args)),
                )
                return make_lookuper(builder)
        else:
            memory = {}

            def wrapper(arg):
                if not memory:
                    memory[object()] = None
                    memory.update(func())

                if silent:
                    return memory.get(arg)
                if arg in memory:
                    return memory[arg]
                raise LookupError(
                    'Failed to look up %s(%s)' % (func.__name__, arg)
                )

        return wraps(func)(wrapper)

    return make_lookuper


make_lookuper = _make_lookuper(False)
silent_lookuper = _make_lookuper(True)
silent_lookuper.__name__ = 'silent_lookuper'