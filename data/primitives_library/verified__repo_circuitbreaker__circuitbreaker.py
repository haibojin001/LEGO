from asyncio import iscoroutinefunction
from datetime import datetime, timedelta, timezone
from functools import wraps
from inspect import isasyncgenfunction, isclass, isgeneratorfunction
from math import ceil, floor
from time import monotonic
from typing import AnyStr, Iterable


STRING_TYPES = (bytes, str)

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


def in_exception_list(*exc_types):
    def predicate(exception_type, exception):
        return issubclass(exception_type, exc_types)

    return predicate


def build_failure_predicate(expected_exception):
    if isclass(expected_exception) and issubclass(expected_exception, Exception):
        return in_exception_list(expected_exception)

    try:
        iter(expected_exception)
        if isinstance(expected_exception, STRING_TYPES):
            raise ValueError(
                "expected_exception cannot be a string. Did you mean name?"
            )
        return in_exception_list(*expected_exception)
    except TypeError:
        if not callable(expected_exception) or isclass(expected_exception):
            raise ValueError("expected_exception does not look like a predicate")
        return expected_exception


class CircuitBreakerError(Exception):
    def __init__(self, circuit_breaker):
        self.circuit_breaker = circuit_breaker


class CircuitBreaker(object):
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 30
    EXPECTED_EXCEPTION = Exception
    FALLBACK_FUNCTION = None

    def __init__(
        self,
        failure_threshold=None,
        recovery_timeout=None,
        expected_exception=None,
        name=None,
        fallback_function=None,
    ):
        self._last_failure = None
        self._failure_count = 0
        self._failure_threshold = failure_threshold or self.FAILURE_THRESHOLD
        self._recovery_timeout = recovery_timeout or self.RECOVERY_TIMEOUT

        if not expected_exception:
            try:
                expected_exception = type(self).__dict__["EXPECTED_EXCEPTION"]
            except KeyError:
                expected_exception = CircuitBreaker.EXPECTED_EXCEPTION

        self.is_failure = build_failure_predicate(expected_exception)
        self._fallback_function = fallback_function or self.FALLBACK_FUNCTION
        self._name = name
        self._state = STATE_CLOSED
        self._opened = monotonic()

    def __call__(self, wrapped):
        return self.decorate(wrapped)

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, _traceback):
        if exc_type and self.is_failure(exc_type, exc_value):
            self._last_failure = exc_value
            self.__call_failed()
        else:
            self.reset()
        return False

    def decorate(self, function):
        if self._name is None:
            try:
                self._name = function.__qualname__
            except AttributeError:
                self._name = function.__name__

        CircuitBreakerMonitor.register(self)

        if iscoroutinefunction(function) or isasyncgenfunction(function):
            return self._decorate_async(function)
        return self._decorate_sync(function)

    def _decorate_sync(self, function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            if self.opened:
                if self.fallback_function:
                    return self.fallback_function(*args, **kwargs)
                raise CircuitBreakerError(self)
            return self.call(function, *args, **kwargs)

        @wraps(function)
        def wrapped_generator(*args, **kwargs):
            if self.opened:
                if self.fallback_function:
                    yield from self.fallback_function(*args, **kwargs)
                    return
                raise CircuitBreakerError(self)
            yield from self.call_generator(function, *args, **kwargs)

        if isgeneratorfunction(function):
            return wrapped_generator
        return wrapped_function

    def _decorate_async(self, function):
        @wraps(function)
        async def wrapped_function(*args, **kwargs):
            if self.opened:
                if self.fallback_function:
                    return await self.fallback_function(*args, **kwargs)
                raise CircuitBreakerError(self)
            return await self.call_async(function, *args, **kwargs)

        @wraps(function)
        async def wrapped_generator(*args, **kwargs):
            if self.opened:
                if self.fallback_function:
                    async for item in self.fallback_function(*args, **kwargs):
                        yield item
                    return
                raise CircuitBreakerError(self)

            async for item in self.call_async_generator(function, *args, **kwargs):
                yield item

        if isasyncgenfunction(function):
            return wrapped_generator
        return wrapped_function

    def call(self, func, *args, **kwargs):
        with self:
            return func(*args, **kwargs)

    def call_generator(self, func, *args, **kwargs):
        with self:
            for item in func(*args, **kwargs):
                yield item

    async def call_async(self, func, *args, **kwargs):
        with self:
            return await func(*args, **kwargs)

    async def call_async_generator(self, func, *args, **kwargs):
        with self:
            async for item in func(*args, **kwargs):
                yield item

    def reset(self):
        self._state = STATE_CLOSED
        self._last_failure = None
        self._failure_count = 0

    def __call_failed(self):
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._state = STATE_OPEN
            self._opened = monotonic()

    @property
    def state(self):
        if self._state == STATE_OPEN and self.open_remaining <= 0:
            return STATE_HALF_OPEN
        return self._state

    @property
    def open_until(self):
        return datetime.now(timezone.utc) + timedelta(seconds=self.open_remaining)

    @property
    def open_remaining(self):
        remaining = (self._opened + self._recovery_timeout) - monotonic()
        if remaining > 0:
            return ceil(remaining)
        return floor(remaining)

    @property
    def opened(self):
        return self.state == STATE_OPEN

    @property
    def closed(self):
        return self.state == STATE_CLOSED

    @property
    def half_open(self):
        return self.state == STATE_HALF_OPEN

    @property
    def name(self):
        return self._name

    @property
    def failure_count(self):
        return self._failure_count

    @property
    def failure_threshold(self):
        return self._failure_threshold

    @property
    def recovery_timeout(self):
        return self._recovery_timeout

    @property
    def last_failure(self):
        return self._last_failure

    @property
    def fallback_function(self):
        return self._fallback_function


class CircuitBreakerMonitor(object):
    circuit_breakers = {}

    @classmethod
    def register(cls, circuit_breaker):
        cls.circuit_breakers[circuit_breaker.name] = circuit_breaker

    @classmethod
    def get(cls, name: AnyStr) -> CircuitBreaker:
        return cls.circuit_breakers.get(name)

    @classmethod
    def get_open(cls) -> Iterable[CircuitBreaker]:
        return [
            circuit_breaker
            for circuit_breaker in cls.circuit_breakers.values()
            if circuit_breaker.opened
        ]

    @classmethod
    def get_closed(cls) -> Iterable[CircuitBreaker]:
        return [
            circuit_breaker
            for circuit_breaker in cls.circuit_breakers.values()
            if circuit_breaker.closed
        ]

    @classmethod
    def all_closed(cls):
        return all(
            circuit_breaker.closed
            for circuit_breaker in cls.circuit_breakers.values()
        )


def circuit(
    failure_threshold=None,
    recovery_timeout=None,
    expected_exception=None,
    name=None,
    fallback_function=None,
):
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name=name,
        fallback_function=fallback_function,
    )