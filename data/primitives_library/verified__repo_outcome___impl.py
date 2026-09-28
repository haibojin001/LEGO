from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Generic,
    NoReturn,
    TypeVar,
    Union,
    overload,
)

import attr

from ._util import AlreadyUsedError, remove_tb_frames

if TYPE_CHECKING:
    from typing_extensions import ParamSpec, final

    ArgsT = ParamSpec("ArgsT")
else:

    def final(func):
        return func


__all__ = ["Error", "Outcome", "Maybe", "Value", "acapture", "capture"]

ValueT = TypeVar("ValueT", covariant=True)
ResultT = TypeVar("ResultT")


@overload
def capture(
    sync_fn: Callable[ArgsT, NoReturn],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Error:
    ...


@overload
def capture(
    sync_fn: Callable[ArgsT, ResultT],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Value[ResultT] | Error:
    ...


def capture(
    sync_fn: Callable[ArgsT, ResultT],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Value[ResultT] | Error:
    try:
        return Value(sync_fn(*args, **kwargs))
    except BaseException as exc:
        return Error(remove_tb_frames(exc, 1))


@overload
async def acapture(
    async_fn: Callable[ArgsT, Awaitable[NoReturn]],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Error:
    ...


@overload
async def acapture(
    async_fn: Callable[ArgsT, Awaitable[ResultT]],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Value[ResultT] | Error:
    ...


async def acapture(
    async_fn: Callable[ArgsT, Awaitable[ResultT]],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Value[ResultT] | Error:
    try:
        return Value(await async_fn(*args, **kwargs))
    except BaseException as exc:
        return Error(remove_tb_frames(exc, 1))


@attr.s(repr=False, init=False, slots=True)
class Outcome(abc.ABC, Generic[ValueT]):
    _unwrapped: bool = attr.ib(default=False, eq=False, init=False)

    def _set_unwrapped(self) -> None:
        if self._unwrapped:
            raise AlreadyUsedError
        object.__setattr__(self, "_unwrapped", True)

    @abc.abstractmethod
    def unwrap(self) -> ValueT:
        ...

    @abc.abstractmethod
    def send(self, gen: Generator[ResultT, ValueT, object]) -> ResultT:
        ...

    @abc.abstractmethod
    async def asend(self, agen: AsyncGenerator[ResultT, ValueT]) -> ResultT:
        ...


@final
@attr.s(frozen=True, repr=False, slots=True)
class Value(Outcome[ValueT], Generic[ValueT]):
    value: ValueT = attr.ib()

    def __repr__(self) -> str:
        return f"Value({self.value!r})"

    def unwrap(self) -> ValueT:
        self._set_unwrapped()
        return self.value

    def send(self, gen: Generator[ResultT, ValueT, object]) -> ResultT:
        self._set_unwrapped()
        return gen.send(self.value)

    async def asend(self, agen: AsyncGenerator[ResultT, ValueT]) -> ResultT:
        self._set_unwrapped()
        return await agen.asend(self.value)


@final
@attr.s(frozen=True, repr=False, slots=True)
class Error(Outcome[NoReturn]):
    error: BaseException = attr.ib(
        validator=attr.validators.instance_of(BaseException)
    )

    def __repr__(self) -> str:
        return f"Error({self.error!r})"

    def unwrap(self) -> NoReturn:
        self._set_unwrapped()
        captured_error = self.error
        try:
            raise captured_error
        finally:
            del captured_error, self

    def send(self, gen: Generator[ResultT, NoReturn, object]) -> ResultT:
        self._set_unwrapped()
        return gen.throw(self.error)

    async def asend(self, agen: AsyncGenerator[ResultT, NoReturn]) -> ResultT:
        self._set_unwrapped()
        return await agen.athrow(self.error)


Maybe = Union[Value[ValueT], Error]