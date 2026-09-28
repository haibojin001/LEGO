import asyncio
import json
import logging
from asyncio import Lock
from collections.abc import AsyncGenerator, Awaitable, Callable
from functools import partial
from http import HTTPStatus
from io import StringIO
from logging import Logger, LoggerAdapter
from typing import Any, Literal, cast, get_args

from anyio import (
    CancelScope,
    create_task_group,
    get_cancelled_exc_class,
    move_on_after,
    sleep,
)
from graphql import DocumentNode, GraphQLSchema, MiddlewareManager
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from ..asgi.handlers import GraphQLHTTPHandler
from ..exceptions import HttpError
from ..format_error import format_error
from ..graphql import (
    ExecutionResult,
    GraphQLError,
    parse_query,
    subscribe,
    validate_data,
)
from ..logger import log_error
from ..subscription_handlers.events import SubscriptionEventType
from ..subscription_handlers.handlers import SubscriptionHandler
from ..types import (
    ErrorFormatter,
    Extensions,
    Middlewares,
    QueryParser,
    QueryValidator,
    RootValue,
    ValidationRules,
)

EVENT_TYPES = Literal["next", "complete"]


class GraphQLServerSentEvent:
    DEFAULT_SEPARATOR = "\r\n"

    def __init__(
        self,
        event: EVENT_TYPES,
        result: ExecutionResult | None = None,
    ):
        assert event in get_args(EVENT_TYPES), f"Invalid event type: {event}"
        self.event = event
        self.result = result
        self.logger = logging.Logger("GraphQLServerSentEvent")

    def __str__(self) -> str:
        buffer = StringIO()
        self._write_to_buffer(buffer, "event", self.event)
        self._write_to_buffer(
            buffer,
            "data",
            self.encode_execution_result()
            if self.event == "next" and self.result
            else "",
        )
        buffer.write(self.DEFAULT_SEPARATOR)
        return buffer.getvalue()

    def _write_to_buffer(
        self, buffer: StringIO, name: str, value: str | None
    ) -> StringIO:
        if value is not None:
            buffer.write(f"{name}: {value}{self.DEFAULT_SEPARATOR}")
        return buffer

    def encode_execution_result(self) -> str:
        payload: dict[str, Any] = {}

        if self.result is not None and self.result.data is not None:
            payload["data"] = self.result.data

        if self.result is not None and self.result.errors is not None:
            payload["errors"] = [format_error(error) for error in self.result.errors]

        return json.dumps(payload)


class ServerSentEventResponse(Response):
    DEFAULT_PING_INTERVAL = 15

    def __init__(
        self,
        *args,
        generator: AsyncGenerator[GraphQLServerSentEvent, Any],
        send_timeout: int | None = None,
        ping_interval: int | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.status_code = HTTPStatus.OK
        self.send_timeout = send_timeout
        self.ping_interval = ping_interval or self.DEFAULT_PING_INTERVAL
        self.body = None

        response_headers: dict[str, str] = {}
        if headers is not None:
            response_headers.update(headers)

        response_headers.setdefault("Cache-Control", "no-cache")
        response_headers.setdefault("Connection", "keep-alive")
        response_headers.setdefault("X-Accel-Buffering", "no")
        response_headers.setdefault("Transfer-Encoding", "chunked")

        self.media_type = "text/event-stream"
        self.init_headers(response_headers)
        self._send_lock = Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with create_task_group() as task_group:

            async def run_and_cancel(
                callback: Callable[[], Awaitable[None]],
            ) -> None:
                await callback()
                task_group.cancel_scope.cancel()

            task_group.start_soon(run_and_cancel, partial(self._ping, send))
            task_group.start_soon(run_and_cancel, partial(self.send_events, send))
            await run_and_cancel(partial(self.listen_for_disconnect, receive))

    async def _ping(self, send: Send) -> None:
        while True:
            await sleep(self.ping_interval)
            async with self._send_lock:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b":\r\n\r\n",
                        "more_body": True,
                    }
                )

    async def send_events(self, send: Send) -> None:
        async with self._send_lock:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )

        try:
            async for event in self.generator:
                async with self._send_lock:
                    with move_on_after(self.send_timeout) as timeout:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": self.encode_event(event),
                                "more_body": True,
                            }
                        )

                    if timeout.cancel_called:
                        raise asyncio.TimeoutError()
        except (get_cancelled_exc_class(),) as error:
            logging.warning(error)
        finally:
            with CancelScope(shield=True):
                async with self._send_lock:
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"",
                            "more_body": False,
                        }
                    )

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break

    def encode_event(self, event: GraphQLServerSentEvent) -> bytes:
        return str(event).encode("utf-8")


class GraphQLSSEHandler(GraphQLHTTPHandler, SubscriptionHandler):
    def __init__(
        self,
        *,
        send_timeout: int | None = None,
        ping_interval: int | None = None,
    ) -> None:
        super().__init__()
        self.send_timeout = send_timeout
        self.ping_interval = ping_interval

    def supports(self, request: Request, data: dict) -> bool:
        accept = request.headers.get("accept", "")
        return "text/event-stream" in accept.lower()

    async def handle(
        self,
        request: Request | Scope,
        data: dict | Receive,
        *args: Any,
        **kwargs: Any,
    ) -> Response | None:
        if isinstance(request, dict) and "type" in request:
            receive = cast(Receive, data)
            send = cast(Send, args[0])
            await GraphQLHTTPHandler.handle(self, request, receive, send)
            return None

        return await self.handle_subscription(
            cast(Request, request),
            cast(dict, data),
            **kwargs,
        )

    async def handle_request(self, request: Request) -> Response:
        if request.method != "POST":
            return await super().handle_request(request)

        try:
            data = await self.extract_data_from_request(request)
        except HttpError:
            return await super().handle_request(request)

        if not self.supports(request, data):
            return await super().handle_request(request)

        context_value = await self.get_context_for_request(request, data)

        return await self.handle_subscription(
            request,
            data,
            schema=self.schema,
            context_value=context_value,
            root_value=self.root_value,
            query_parser=self.query_parser,
            query_validator=self.query_validator,
            validation_rules=self.validation_rules,
            debug=self.debug,
            introspection=self.introspection,
            logger=self.logger,
            error_formatter=self.error_formatter,
        )

    async def handle_subscription(
        self,
        request: Request,
        data: dict,
        *,
        schema: GraphQLSchema,
        context_value: Any,
        root_value: RootValue | None,
        query_parser: QueryParser | None,
        query_validator: QueryValidator | None,
        validation_rules: ValidationRules | None,
        debug: bool,
        introspection: bool,
        logger: None | str | Logger | LoggerAdapter,
        error_formatter: ErrorFormatter,
    ) -> Response:
        return ServerSentEventResponse(
            generator=self.generate_sse_events(
                data,
                schema=schema,
                context_value=context_value,
                root_value=root_value,
                query_parser=query_parser,
                query_validator=query_validator,
                validation_rules=validation_rules,
                debug=debug,
                introspection=introspection,
                logger=logger,
                error_formatter=error_formatter,
            ),
            send_timeout=self.send_timeout,
            ping_interval=self.ping_interval,
        )

    async def generate_sse_events(
        self,
        data: dict,
        *,
        schema: GraphQLSchema,
        context_value: Any,
        root_value: RootValue | None,
        query_parser: QueryParser | None,
        query_validator: QueryValidator | None,
        validation_rules: ValidationRules | None,
        debug: bool,
        introspection: bool,
        logger: None | str | Logger | LoggerAdapter,
        error_formatter: ErrorFormatter,
    ) -> AsyncGenerator[GraphQLServerSentEvent, None]:
        async for event in SubscriptionHandler.generate_events(
            self,
            data,
            schema=schema,
            context_value=context_value,
            root_value=root_value,
            query_parser=query_parser,
            query_validator=query_validator,
            query_document=None,
            validation_rules=validation_rules,
            debug=debug,
            introspection=introspection,
            logger=logger,
            error_formatter=error_formatter,
        ):
            if event.event_type == SubscriptionEventType.DATA:
                yield GraphQLServerSentEvent("next", event.result)
            elif event.event_type == SubscriptionEventType.ERROR:
                yield GraphQLServerSentEvent("next", event.result)
                yield GraphQLServerSentEvent("complete")
            elif event.event_type == SubscriptionEventType.COMPLETE:
                yield GraphQLServerSentEvent("complete")