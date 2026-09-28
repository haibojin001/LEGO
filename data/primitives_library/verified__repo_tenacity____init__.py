import functools
import sys
import typing as t

import tenacity
from tenacity import (
    AttemptManager,
    BaseRetrying,
    DoAttempt,
    DoSleep,
    RetryCallState,
    RetryError,
    _RetryDecorated,
    _utils,
    after_nothing,
    before_nothing,
)
from tenacity._utils import override

from .retry import (
    RetryBaseT,
    retry_all,
    retry_any,
    retry_if_exception,
    retry_if_result,
)

if t.TYPE_CHECKING:
    from tenacity.retry import RetryBaseT as SyncRetryBaseT
    from tenacity.stop import StopBaseT
    from tenacity.wait import WaitBaseT

WrappedFnReturnT = t.TypeVar("WrappedFnReturnT")
WrappedFn = t.TypeVar("WrappedFn", bound=t.Callable[..., t.Awaitable[t.Any]])
P = t.ParamSpec("P")
R = t.TypeVar("R")


def _portable_async_sleep(seconds: float) -> t.Awaitable[None]:
    if "trio" in sys.modules:
        import sniffio
        import trio

        if sniffio.current_async_library() == "trio":
            return trio.sleep(seconds)

    import asyncio

    return asyncio.sleep(seconds)


class AsyncRetrying(BaseRetrying):
    def __init__(
        self,
        sleep: t.Callable[
            [int | float], t.Awaitable[None] | None
        ] = _portable_async_sleep,
        stop: "StopBaseT" = tenacity.stop.stop_never,
        wait: "WaitBaseT" = tenacity.wait.wait_none(),
        retry: "SyncRetryBaseT | RetryBaseT" = tenacity.retry_if_exception_type(),
        before: t.Callable[
            ["RetryCallState"], t.Awaitable[None] | None
        ] = before_nothing,
        after: t.Callable[
            ["RetryCallState"], t.Awaitable[None] | None
        ] = after_nothing,
        before_sleep: t.Callable[
            ["RetryCallState"], t.Awaitable[None] | None
        ]
        | None = None,
        reraise: bool = False,
        retry_error_cls: type["RetryError"] = RetryError,
        retry_error_callback: t.Callable[
            ["RetryCallState"], t.Any | t.Awaitable[t.Any]
        ]
        | None = None,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            sleep=sleep,
            stop=stop,
            wait=wait,
            retry=retry,
            before=before,
            after=after,
            before_sleep=before_sleep,
            reraise=reraise,
            retry_error_cls=retry_error_cls,
            retry_error_callback=retry_error_callback,
            name=name,
            enabled=enabled,
        )

    @override
    async def __call__(
        self, fn: WrappedFn, *args: t.Any, **kwargs: t.Any
    ) -> WrappedFnReturnT:
        is_async = _utils.is_coroutine_callable(fn)

        if not self.enabled:
            if is_async:
                return await fn(*args, **kwargs)
            return fn(*args, **kwargs)

        self.begin()
        retry_state = RetryCallState(
            retry_object=self,
            fn=fn,
            args=args,
            kwargs=kwargs,
        )

        while True:
            action = await self.iter(retry_state=retry_state)

            if isinstance(action, DoAttempt):
                try:
                    if is_async:
                        result = await fn(*args, **kwargs)
                    else:
                        result = fn(*args, **kwargs)
                except BaseException:
                    retry_state.set_exception(sys.exc_info())
                else:
                    retry_state.set_result(result)
            elif isinstance(action, DoSleep):
                retry_state.prepare_for_next_attempt()
                await self.sleep(action)
            else:
                return action

    @override
    def _add_action_func(self, fn: t.Callable[..., t.Any]) -> None:
        self.iter_state.actions.append(_utils.wrap_to_async_func(fn))

    @override
    async def _run_retry(self, retry_state: RetryCallState) -> None:
        self.iter_state.retry_run_result = await _utils.wrap_to_async_func(self.retry)(
            retry_state
        )

    @override
    async def _run_wait(self, retry_state: RetryCallState) -> None:
        if not self.wait:
            retry_state.upcoming_sleep = 0.0
        else:
            retry_state.upcoming_sleep = await _utils.wrap_to_async_func(self.wait)(
                retry_state
            )

    @override
    async def _run_stop(self, retry_state: RetryCallState) -> None:
        self.statistics["delay_since_first_attempt"] = retry_state.seconds_since_start
        self.iter_state.stop_run_result = await _utils.wrap_to_async_func(self.stop)(
            retry_state
        )

    @override
    async def iter(self, retry_state: RetryCallState) -> DoAttempt | DoSleep | t.Any:
        self._begin_iter(retry_state)
        result = None
        for action in self.iter_state.actions:
            result = await action(retry_state)
        return result

    @override
    def __iter__(self) -> t.Generator[AttemptManager, None, None]:
        raise TypeError("AsyncRetrying object is not iterable")

    def __aiter__(self) -> "AsyncRetrying":
        if not self.enabled:
            self._retry_state = RetryCallState(self, fn=None, args=(), kwargs={})
            self._disabled_iter_done = False
            return self

        self.begin()
        self._retry_state = RetryCallState(self, fn=None, args=(), kwargs={})
        return self

    async def __anext__(self) -> AttemptManager:
        if not self.enabled:
            if self._disabled_iter_done:
                outcome = self._retry_state.outcome
                if outcome is not None and outcome.failed:
                    raise outcome.exception()
                raise StopAsyncIteration

            self._disabled_iter_done = True
            return AttemptManager(retry_state=self._retry_state)

        while True:
            action = await self.iter(retry_state=self._retry_state)

            if action is None:
                raise StopAsyncIteration

            if isinstance(action, DoAttempt):
                return AttemptManager(retry_state=self._retry_state)

            if isinstance(action, DoSleep):
                self._retry_state.prepare_for_next_attempt()
                await self.sleep(action)
            else:
                raise StopAsyncIteration

    @override
    def wraps(self, fn: t.Callable[P, R]) -> _RetryDecorated[P, R]:
        wrapped = super().wraps(fn)

        @functools.wraps(
            fn,
            functools.WRAPPER_ASSIGNMENTS + ("__defaults__", "__kwdefaults__"),
        )
        async def async_wrapped(*args: t.Any, **kwargs: t.Any) -> t.Any:
            if not self.enabled:
                return await fn(*args, **kwargs)

            copy = self.copy()
            statistics = async_wrapped.statistics
            statistics.clear()
            copy._local.statistics = statistics
            self._local.statistics = statistics
            return await copy(fn, *args, **kwargs)

        async_wrapped.retry = self
        async_wrapped.retry_with = wrapped.retry_with
        async_wrapped.statistics = {}
        return async_wrapped