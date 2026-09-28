import abc
import re
import typing

from tenacity._utils import override

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState


class retry_base(abc.ABC):
    """Base type for retry-decision policies."""

    @abc.abstractmethod
    def __call__(self, retry_state: "RetryCallState") -> bool:
        pass

    def __and__(self, other: "RetryBaseT") -> "retry_all":
        if isinstance(other, retry_base):
            return other.__rand__(self)
        if isinstance(self, retry_all):
            return retry_all(*self.retries, other)
        return retry_all(self, other)

    def __rand__(self, other: "RetryBaseT") -> "retry_all":
        if isinstance(other, retry_all):
            return retry_all(*other.retries, self)
        return retry_all(other, self)

    def __or__(self, other: "RetryBaseT") -> "retry_any":
        if isinstance(other, retry_base):
            return other.__ror__(self)
        if isinstance(self, retry_any):
            return retry_any(*self.retries, other)
        return retry_any(self, other)

    def __ror__(self, other: "RetryBaseT") -> "retry_any":
        if isinstance(other, retry_any):
            return retry_any(*other.retries, self)
        return retry_any(other, self)


RetryBaseT = retry_base | typing.Callable[["RetryCallState"], bool]


class _retry_never(retry_base):
    """A retry policy which never asks for another attempt."""

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return False


retry_never = _retry_never()


class _retry_always(retry_base):
    """A retry policy which always asks for another attempt."""

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return True


retry_always = _retry_always()


class retry_if_exception(retry_base):
    """Retry when a raised exception satisfies a supplied test."""

    def __init__(self, predicate: typing.Callable[[BaseException], bool]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            if exception is None:
                raise RuntimeError("outcome failed but the exception is None")
            return self.predicate(exception)

        return False


class retry_if_exception_type(retry_if_exception):
    """Retry for exceptions belonging to one or more specified classes."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(self._check)

    def _check(self, exception: BaseException) -> bool:
        return isinstance(exception, self.exception_types)


class retry_if_not_exception_type(retry_if_exception):
    """Retry unless the exception belongs to one of the specified classes."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(self._check)

    def _check(self, exception: BaseException) -> bool:
        return not isinstance(exception, self.exception_types)


class retry_unless_exception_type(retry_if_exception):
    """Continue retrying until an exception of a chosen class is raised."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(self._check)

    def _check(self, exception: BaseException) -> bool:
        return not isinstance(exception, self.exception_types)

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return True

        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError("outcome failed but the exception is None")
        return self.predicate(exception)


class retry_if_exception_cause_type(retry_base):
    """Retry when a cause in an exception's explicit cause chain has a given type."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_cause_types = exception_types

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__ called before outcome was set")

        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            visited: set[int] = set()

            while exception is not None and id(exception) not in visited:
                visited.add(id(exception))
                if isinstance(exception.__cause__, self.exception_cause_types):
                    return True
                exception = exception.__cause__

        return False


class retry_if_result(retry_base):
    """Retry when a successful return value satisfies a predicate."""

    def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return self.predicate(retry_state.outcome.result())

        return False


class retry_if_not_result(retry_base):
    """Retry when a successful return value does not satisfy a predicate."""

    def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return not self.predicate(retry_state.outcome.result())

        return False


class retry_if_exception_message(retry_if_exception):
    """Retry when an exception text is equal to, or matches, a supplied value."""

    def __init__(
        self,
        message: str | None = None,
        match: str | re.Pattern[str] | None = None,
    ) -> None:
        if message is not None and match is not None:
            raise TypeError(
                f"{self.__class__.__name__}() takes either 'message' or 'match', not both"
            )

        if message is None and match is None:
            raise TypeError(
                f"{self.__class__.__name__}() missing 1 required argument 'message' or 'match'"
            )

        self.message = message
        self.match: re.Pattern[str] | None = (
            re.compile(match) if match is not None else None
        )
        super().__init__(self._check)

    def _check(self, exception: BaseException) -> bool:
        if self.message is not None:
            return str(exception) == self.message
        assert self.match is not None
        return bool(self.match.match(str(exception)))


class retry_if_not_exception_message(retry_if_exception_message):
    """Retry until an exception text is equal to, or matches, a supplied value."""

    @override
    def _check(self, exception: BaseException) -> bool:
        return not super()._check(exception)

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return True

        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError("outcome failed but the exception is None")
        return self.predicate(exception)


class retry_any(retry_base):
    """Retry if at least one contained retry policy requests it."""

    def __init__(self, *retries: RetryBaseT) -> None:
        self.retries = retries

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return any(retry(retry_state) for retry in self.retries)


class retry_all(retry_base):
    """Retry only if every contained retry policy requests it."""

    def __init__(self, *retries: RetryBaseT) -> None:
        self.retries = retries

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return all(retry(retry_state) for retry in self.retries)