from collections.abc import Hashable
from contextlib import suppress
from datetime import datetime, timedelta
import time
import threading

from .decorators import decorator, wraps, get_argnames, arggetter, contextmanager

try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, *exc_info):
            return None


__all__ = [
    'raiser', 'ignore', 'silent', 'suppress', 'nullcontext', 'reraise', 'retry',
    'fallback', 'limit_error_rate', 'ErrorRateExceeded', 'throttle',
    'post_processing', 'collecting', 'joining', 'once', 'once_per',
    'once_per_args', 'wrap_with'
]


def _is_exception_type(value):
    return isinstance(value, type) and issubclass(value, BaseException)


def _ensure_exceptable(errors):
    if _is_exception_type(errors):
        return errors
    return tuple(errors)


def raiser(exception_or_class=Exception, *args, **kwargs):
    if isinstance(exception_or_class, str):
        exception_or_class = Exception(exception_or_class)

    def raise_it(*call_args, **call_kwargs):
        if args or kwargs:
            raise exception_or_class(*args, **kwargs)
        raise exception_or_class

    return raise_it


def ignore(errors, default=None):
    handled = _ensure_exceptable(errors)

    def apply(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except handled:
                return default
        return wrapped

    return apply


def silent(func):
    return ignore(Exception)(func)


@contextmanager
def reraise(errors, into):
    handled = _ensure_exceptable(errors)
    try:
        yield
    except handled as error:
        target = into(error) if callable(into) and not _is_exception_type(into) else into
        raise target from error


@decorator
def retry(call, tries, errors=Exception, timeout=0, filter_errors=None):
    handled = _ensure_exceptable(errors)

    for attempt in range(tries):
        try:
            return call()
        except handled as error:
            if filter_errors is not None and not filter_errors(error):
                raise

            if attempt + 1 == tries:
                raise

            delay = timeout(attempt) if callable(timeout) else timeout
            if delay > 0:
                time.sleep(delay)


def fallback(*approaches):
    for approach in approaches:
        if callable(approach):
            func, errors = approach, Exception
        else:
            func, errors = approach

        try:
            return func()
        except _ensure_exceptable(errors):
            continue


class ErrorRateExceeded(Exception):
    pass


def limit_error_rate(fails, timeout, exception=ErrorRateExceeded):
    duration = timedelta(seconds=timeout) if isinstance(timeout, int) else timeout

    def apply(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            blocked_at = wrapped.blocked
            if blocked_at is not None:
                if datetime.now() - blocked_at < duration:
                    raise exception
                wrapped.blocked = None

            try:
                result = func(*args, **kwargs)
            except BaseException:
                wrapped.fails += 1
                if wrapped.fails >= fails:
                    wrapped.blocked = datetime.now()
                raise
            else:
                wrapped.fails = 0
                return result

        wrapped.fails = 0
        wrapped.blocked = None
        return wrapped

    return apply


def throttle(period):
    seconds = period.total_seconds() if isinstance(period, timedelta) else period

    def apply(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            current = time.time()
            if wrapped.blocked_until > current:
                return None

            wrapped.blocked_until = current + seconds
            return func(*args, **kwargs)

        wrapped.blocked_until = 0
        return wrapped

    return apply


@decorator
def post_processing(call, func):
    return func(call())


collecting = post_processing(list)
collecting.__name__ = 'collecting'
collecting.__doc__ = 'Transforms a generator into list returning function.'


@decorator
def joining(call, sep):
    return sep.join(map(sep.__class__, call()))


def once_per(*argnames):
    def apply(func):
        lock = threading.Lock()
        hashable_values = set()
        unhashable_values = []
        get_value = arggetter(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            with lock:
                values = tuple(get_value(name, args, kwargs) for name in argnames)

                if isinstance(values, Hashable):
                    seen = hashable_values
                    remember = hashable_values.add
                else:
                    seen = unhashable_values
                    remember = unhashable_values.append

                if values not in seen:
                    remember(values)
                    return func(*args, **kwargs)

        return wrapped

    return apply


once = once_per()
once.__doc__ = 'Let function execute once, noop all subsequent calls.'


def once_per_args(func):
    return once_per(*get_argnames(func))(func)


@decorator
def wrap_with(call, ctx):
    with ctx:
        return call()