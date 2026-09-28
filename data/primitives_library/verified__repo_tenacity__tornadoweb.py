import sys
import typing

from tornado import gen

from tenacity import BaseRetrying, DoAttempt, DoSleep, RetryCallState
from tenacity._utils import override

if typing.TYPE_CHECKING:
    from tornado.concurrent import Future

_RetValT = typing.TypeVar("_RetValT")


class TornadoRetrying(BaseRetrying):
    sleep: typing.Callable[..., "Future[None]"]

    def __init__(
        self,
        sleep: "typing.Callable[[float], Future[None]]" = gen.sleep,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sleep = sleep

    @gen.coroutine
    @override
    def __call__(
        self,
        fn: "typing.Callable[..., typing.Generator[typing.Any, typing.Any, _RetValT] | Future[_RetValT]]",
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> "typing.Generator[typing.Any, typing.Any, _RetValT]":
        self.begin()
        retry_state = RetryCallState(
            retry_object=self,
            fn=fn,
            args=args,
            kwargs=kwargs,
        )

        while True:
            action = self.iter(retry_state=retry_state)

            if isinstance(action, DoAttempt):
                try:
                    result = yield fn(*args, **kwargs)
                except BaseException:
                    retry_state.set_exception(sys.exc_info())
                else:
                    retry_state.set_result(result)
            elif isinstance(action, DoSleep):
                retry_state.prepare_for_next_attempt()
                yield self.sleep(action)
            else:
                raise gen.Return(action)