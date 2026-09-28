import abc
import math
import random
import typing
import warnings

from tenacity import _utils
from tenacity._utils import override

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState


class wait_base(abc.ABC):
    """Abstract superclass for retry delay policies."""

    @abc.abstractmethod
    def __call__(self, retry_state: "RetryCallState") -> float:
        pass

    def __add__(self, other: "wait_base") -> "wait_combine":
        return wait_combine(self, other)

    def __radd__(self, other: "WaitBaseT | int") -> "wait_combine | wait_base":
        if isinstance(other, int):
            if other == 0:
                return self
            return NotImplemented
        return wait_combine(self, other)


WaitBaseT = wait_base | typing.Callable[["RetryCallState"], float | int]


class wait_fixed(wait_base):
    """Return the same pause duration for every retry."""

    def __init__(self, wait: _utils.time_unit_type) -> None:
        self.wait_fixed = _utils.to_seconds(wait)

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        return self.wait_fixed


class wait_none(wait_fixed):
    """Return zero delay for every retry."""

    def __init__(self) -> None:
        super().__init__(0)


class wait_random(wait_base):
    """Choose a random delay between the configured bounds."""

    def __init__(
        self, min: _utils.time_unit_type = 0, max: _utils.time_unit_type = 1
    ) -> None:
        self.wait_random_min = _utils.to_seconds(min)
        self.wait_random_max = _utils.to_seconds(max)

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        return self.wait_random_min + random.random() * (
            self.wait_random_max - self.wait_random_min
        )


class wait_combine(wait_base):
    """Add the delay returned by each configured wait policy."""

    def __init__(self, *strategies: "WaitBaseT") -> None:
        self.wait_funcs = strategies

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        return float(sum(strategy(retry_state) for strategy in self.wait_funcs))


class wait_chain(wait_base):
    """Use each policy in sequence, retaining the last policy indefinitely."""

    def __init__(self, *strategies: wait_base) -> None:
        if not strategies:
            raise ValueError("wait_chain() requires at least one strategy")
        self.strategies = strategies

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        index = min(max(retry_state.attempt_number, 1), len(self.strategies))
        return self.strategies[index - 1](retry_state=retry_state)


class wait_exception(wait_base):
    """Obtain the delay by applying a predicate to the previous exception."""

    def __init__(self, predicate: typing.Callable[[BaseException], float]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        error = retry_state.outcome.exception()
        if error is None:
            raise RuntimeError("outcome failed but the exception is None")
        return self.predicate(error)


class wait_incrementing(wait_base):
    """Increase the delay by a fixed amount on each successive attempt."""

    def __init__(
        self,
        start: _utils.time_unit_type = 0,
        increment: _utils.time_unit_type = 100,
        max: _utils.time_unit_type = _utils.MAX_WAIT,
    ) -> None:
        self.start = _utils.to_seconds(start)
        self.increment = _utils.to_seconds(increment)
        self.max = _utils.to_seconds(max)

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        delay = self.start + self.increment * (retry_state.attempt_number - 1)
        return max(0, min(delay, self.max))


class wait_exponential(wait_base):
    """Return a capped exponential retry delay without random jitter."""

    def __init__(
        self,
        multiplier: float = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,
        exp_base: float = 2,
        min: _utils.time_unit_type = 0,
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        exponent = retry_state.attempt_number - 1

        if (
            self.multiplier > 0
            and self.max > 0
            and self.exp_base > 1
            and math.isfinite(self.max)
            and exponent
            > math.log(self.max, self.exp_base)
            - math.log(self.multiplier, self.exp_base)
        ):
            return max(max(0, self.min), self.max)

        try:
            value = self.multiplier * self.exp_base**exponent
        except OverflowError:
            return self.max

        return max(max(0, self.min), min(value, self.max))


class wait_random_exponential(wait_exponential):
    """Use exponential backoff with a randomly selected delay in its window."""

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        upper = super().__call__(retry_state=retry_state)
        return random.uniform(self.min, upper)


class wait_exponential_jitter(wait_base):
    """Use exponential backoff and add a bounded random jitter component."""

    def __init__(
        self,
        initial: float = 1,
        max: float = _utils.MAX_WAIT,
        exp_base: float = 2,
        jitter: float = 1,
    ) -> None:
        self.initial = initial
        self.max = max
        self.exp_base = exp_base
        self.jitter = jitter

    @override
    def __call__(self, retry_state: "RetryCallState") -> float:
        noise = random.uniform(0, self.jitter)
        try:
            delay = self.initial * self.exp_base ** (
                retry_state.attempt_number - 1
            ) + noise
        except OverflowError:
            delay = self.max
        return min(delay, self.max)