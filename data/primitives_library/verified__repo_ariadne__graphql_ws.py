import asyncio
from collections.abc import AsyncGenerator
from inspect import isawaitable
from typing import Any, cast

from graphql import DocumentNode, GraphQLError
from graphql.language import OperationType
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from ...exceptions import WebSocketConnectionError
from ...graphql import parse_query, subscribe, validate_data
from ...logger import log_error
from ...types import Operation
from ...utils import get_operation_type
from .base import GraphQLWebsocketHandlerBase


class GraphQLWSHandler(GraphQLWebsocketHandlerBase):
    """Handler implementing subscriptions-transport-ws graphql-ws protocol."""

    keepalive: float | None

    GQL_CONNECTION_INIT = "connection_init"
    GQL_CONNECTION_ACK = "connection_ack"
    GQL_CONNECTION_ERROR = "connection_error"
    GQL_CONNECTION_KEEP_ALIVE = "ka"
    GQL_CONNECTION_TERMINATE = "connection_terminate"
    GQL_START = "start"
    GQL_DATA = "data"
    GQL_ERROR = "error"
    GQL_COMPLETE = "complete"
    GQL_STOP = "stop"

    def __init__(
        self,
        *args,
        keepalive: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.keepalive = keepalive

    async def handle(self, scope: Scope, receive: Receive, send: Send):
        websocket = WebSocket(scope=scope, receive=receive, send=send)
        await self.handle_websocket(websocket)

    async def handle_websocket(self, websocket: WebSocket):
        operations: dict[str, Operation] = {}
        connection_acknowledged = False

        await websocket.accept("graphql-ws")

        try:
            while WebSocketState.DISCONNECTED not in (
                websocket.client_state,
                websocket.application_state,
            ):
                message = await websocket.receive_json()
                message_type = message.get("type")

                if (
                    message_type == self.GQL_START
                    and not connection_acknowledged
                ):
                    await websocket.close(code=4401)
                    break

                await self.handle_websocket_message(websocket, message, operations)

                if message_type == self.GQL_CONNECTION_INIT:
                    connection_acknowledged = WebSocketState.DISCONNECTED not in (
                        websocket.client_state,
                        websocket.application_state,
                    )
        except WebSocketDisconnect:
            pass
        finally:
            for operation_id, operation in list(operations.items()):
                await self.stop_websocket_operation(websocket, operation)
                del operations[operation_id]

            try:
                if self.on_disconnect:
                    result = self.on_disconnect(websocket)
                    if result and isawaitable(result):
                        await result
            except Exception as error:
                if not isinstance(error, GraphQLError):
                    error = GraphQLError(str(error), original_error=error)
                log_error(error, self.logger)

    async def handle_websocket_message(
        self,
        websocket: WebSocket,
        message: dict,
        operations: dict[str, Operation],
    ):
        operation_id = cast(str, message.get("id"))
        message_type = cast(str, message.get("type"))

        if message_type == self.GQL_CONNECTION_INIT:
            await self.handle_websocket_connection_init_message(websocket, message)
        elif message_type == self.GQL_CONNECTION_TERMINATE:
            await self.handle_websocket_connection_terminate_message(websocket)
        elif message_type == self.GQL_START:
            await self.process_single_message(
                websocket, message.get("payload"), operation_id, operations
            )
        elif message_type == self.GQL_STOP:
            if operation_id in operations:
                await self.stop_websocket_operation(websocket, operations[operation_id])
                del operations[operation_id]

    async def process_single_message(
        self,
        websocket: WebSocket,
        data: Any,
        operation_id: str,
        operations: dict[str, Operation],
    ) -> None:
        validate_data(data)
        context_value = await self.get_context_for_request(websocket, data)

        try:
            query_document = parse_query(context_value, self.query_parser, data)
        except GraphQLError as error:
            log_error(error, self.logger)
            await websocket.send_json(
                {
                    "type": self.GQL_ERROR,
                    "id": operation_id,
                    "payload": self.error_formatter(error, self.debug),
                }
            )
            return

        operation_type = get_operation_type(query_document, data.get("operationName"))

        if operation_type == OperationType.SUBSCRIPTION:
            await self.start_websocket_operation(
                websocket,
                data,
                context_value,
                query_document,
                operation_id,
                operations,
            )
        else:
            if self.http_handler is None:
                raise TypeError(
                    "http_handler is not set, call configure method to initialize it"
                )

            _, result = await self.http_handler.execute_graphql_query(websocket, data)

            await websocket.send_json(
                {
                    "type": self.GQL_DATA,
                    "id": operation_id,
                    "payload": result,
                }
            )
            await websocket.send_json(
                {
                    "type": self.GQL_COMPLETE,
                    "id": operation_id,
                }
            )

    async def handle_websocket_connection_init_message(
        self,
        websocket: WebSocket,
        message: dict,
    ):
        if self.on_connect:
            try:
                result = self.on_connect(websocket, message.get("payload"))
                if result and isawaitable(result):
                    await result
            except WebSocketConnectionError as error:
                await websocket.send_json(
                    {
                        "type": self.GQL_CONNECTION_ERROR,
                        "payload": error.payload,
                    }
                )
                await websocket.close()
                return

        await websocket.send_json({"type": self.GQL_CONNECTION_ACK})

        if self.keepalive:
            asyncio.ensure_future(self.keep_websocket_alive(websocket))

    async def handle_websocket_connection_terminate_message(
        self,
        websocket: WebSocket,
    ):
        await websocket.close()

    async def start_websocket_operation(
        self,
        websocket: WebSocket,
        data: Any,
        context_value: Any,
        query_document: DocumentNode,
        operation_id: str,
        operations: dict[str, Operation],
    ):
        success, results = await subscribe(
            self.schema,
            data,
            context_value=context_value,
            root_value=self.root_value,
            query_document=query_document,
            validation_rules=self.validation_rules,
            debug=self.debug,
            introspection=self.introspection,
            logger=self.logger,
            error_formatter=self.error_formatter,
        )

        if success:
            if isinstance(results, AsyncGenerator):
                operation = Operation(
                    operation_id,
                    data.get("operationName"),
                    results,
                )
                operations[operation_id] = operation
                asyncio.ensure_future(self.observe_async_results(websocket, operation))
            else:
                await websocket.send_json(
                    {
                        "type": self.GQL_DATA,
                        "id": operation_id,
                        "payload": results,
                    }
                )
                await websocket.send_json(
                    {
                        "type": self.GQL_COMPLETE,
                        "id": operation_id,
                    }
                )
        else:
            await websocket.send_json(
                {
                    "type": self.GQL_ERROR,
                    "id": operation_id,
                    "payload": results,
                }
            )

    async def stop_websocket_operation(
        self,
        websocket: WebSocket,
        operation: Operation,
    ):
        await operation.generator.aclose()

    async def observe_async_results(
        self,
        websocket: WebSocket,
        operation: Operation,
    ):
        async for result in operation.generator:
            await websocket.send_json(
                {
                    "type": self.GQL_DATA,
                    "id": operation.id,
                    "payload": result,
                }
            )

        await websocket.send_json(
            {
                "type": self.GQL_COMPLETE,
                "id": operation.id,
            }
        )

    async def keep_websocket_alive(self, websocket: WebSocket):
        while WebSocketState.DISCONNECTED not in (
            websocket.client_state,
            websocket.application_state,
        ):
            await asyncio.sleep(self.keepalive)
            await websocket.send_json({"type": self.GQL_CONNECTION_KEEP_ALIVE})