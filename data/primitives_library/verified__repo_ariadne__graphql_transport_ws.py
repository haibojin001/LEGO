import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from datetime import timedelta
from inspect import isawaitable
from typing import Any, cast

from graphql import GraphQLError
from graphql.language import OperationType
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from ...graphql import parse_query, subscribe, validate_data
from ...logger import log_error
from ...types import ExecutionResult, Operation
from ...utils import get_operation_type
from .base import GraphQLWebsocketHandlerBase


class ClientContext:
    def __init__(self) -> None:
        self.connection_acknowledged: bool = False
        self.connection_init_timeout_task: asyncio.Task | None = None
        self.connection_init_received: bool = False
        self.operations: dict[str, Operation] = {}
        self.operation_tasks: dict[str, asyncio.Task] = {}
        self.websocket: WebSocket


class GraphQLTransportWSHandler(GraphQLWebsocketHandlerBase):
    GQL_CONNECTION_INIT = "connection_init"
    GQL_CONNECTION_ACK = "connection_ack"
    GQL_PING = "ping"
    GQL_PONG = "pong"
    GQL_SUBSCRIBE = "subscribe"
    GQL_NEXT = "next"
    GQL_ERROR = "error"
    GQL_COMPLETE = "complete"

    def __init__(
        self,
        *args,
        connection_init_wait_timeout: timedelta = timedelta(minutes=1),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.connection_init_wait_timeout = connection_init_wait_timeout

    async def handle(self, scope: Scope, receive: Receive, send: Send):
        websocket = WebSocket(scope=scope, receive=receive, send=send)
        await self.handle_websocket(websocket)

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept("graphql-transport-ws")

        client_context = ClientContext()
        client_context.websocket = websocket
        client_context.connection_init_timeout_task = asyncio.create_task(
            self.handle_connection_init_timeout(websocket, client_context)
        )

        try:
            while WebSocketState.DISCONNECTED not in (
                websocket.client_state,
                websocket.application_state,
            ):
                message = await websocket.receive_json()
                await self.handle_websocket_message(websocket, message, client_context)
        except WebSocketDisconnect:
            pass
        finally:
            for operation_id in list(client_context.operations.keys()):
                await self.stop_websocket_operation(
                    websocket, operation_id, client_context
                )

            timeout_task = client_context.connection_init_timeout_task
            if timeout_task:
                timeout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await timeout_task

            try:
                if self.on_disconnect:
                    result = self.on_disconnect(websocket)
                    if result and isawaitable(result):
                        await result
            except Exception as error:
                if not isinstance(error, GraphQLError):
                    error = GraphQLError(str(error), original_error=error)
                log_error(error, self.logger)

    async def handle_connection_init_timeout(
        self, websocket: WebSocket, client_context: ClientContext
    ):
        await asyncio.sleep(
            delay=self.connection_init_wait_timeout.total_seconds()
        )

        if client_context.connection_init_received:
            return

        if WebSocketState.DISCONNECTED not in (
            websocket.client_state,
            websocket.application_state,
        ):
            await websocket.close(code=4408)

    async def handle_websocket_message(
        self,
        websocket: WebSocket,
        message: dict,
        client_context: ClientContext,
    ):
        operation_id = cast(str, message.get("id"))
        message_type = cast(str, message.get("type"))

        if message_type == self.GQL_CONNECTION_INIT:
            await self.handle_websocket_connection_init_message(
                websocket, message, client_context
            )
        elif message_type == self.GQL_PING:
            await self.handle_websocket_ping_message(websocket, client_context)
        elif message_type == self.GQL_PONG:
            await self.handle_websocket_pong_message(websocket, client_context)
        elif message_type == self.GQL_COMPLETE:
            await self.handle_websocket_complete_message(
                websocket, operation_id, client_context
            )
        elif message_type == self.GQL_SUBSCRIBE:
            await self.handle_websocket_subscribe(
                websocket, message.get("payload"), operation_id, client_context
            )
        else:
            await self.handle_websocket_invalid_type(websocket)

    async def handle_websocket_connection_init_message(
        self,
        websocket: WebSocket,
        message: dict,
        client_context: ClientContext,
    ):
        if client_context.connection_init_received:
            await websocket.close(code=4429)
            return

        client_context.connection_init_received = True

        try:
            if self.on_connect:
                result = self.on_connect(websocket, message.get("payload"))
                if result and isawaitable(result):
                    await result

            await websocket.send_json({"type": self.GQL_CONNECTION_ACK})
            client_context.connection_acknowledged = True
        except Exception as error:
            log_error(error, self.logger)
            await websocket.close()

    async def handle_websocket_ping_message(
        self,
        websocket: WebSocket,
        client_context: ClientContext,
    ):
        await websocket.send_json({"type": self.GQL_PONG})

    async def handle_websocket_pong_message(
        self,
        websocket: WebSocket,
        client_context: ClientContext,
    ):
        return None

    async def handle_websocket_complete_message(
        self,
        websocket: WebSocket,
        operation_id: str,
        client_context: ClientContext,
    ):
        if operation_id in client_context.operations:
            await self.stop_websocket_operation(
                websocket, operation_id, client_context
            )

    async def handle_websocket_invalid_type(self, websocket: WebSocket):
        await websocket.close(code=4400)

    async def handle_websocket_subscribe(
        self,
        websocket: WebSocket,
        data: Any,
        operation_id: str,
        client_context: ClientContext,
    ):
        if not client_context.connection_acknowledged:
            await websocket.close(code=4401)
            return

        if operation_id in client_context.operations:
            await websocket.close(code=4409)
            return

        if not isinstance(operation_id, str) or not isinstance(data, dict):
            await websocket.close(code=4400)
            return

        query = data.get("query")
        operation_name = data.get("operationName")
        variables = data.get("variables")

        if not isinstance(query, str):
            await websocket.close(code=4400)
            return

        try:
            document = parse_query(query, self.query_parser)
            validation_errors = validate_data(
                self.schema,
                document,
                self.validation_rules,
                self.require_query,
                self.introspection,
            )
        except GraphQLError as error:
            await self.send_websocket_error(websocket, operation_id, [error])
            return
        except Exception as error:
            graphql_error = GraphQLError(str(error), original_error=error)
            log_error(graphql_error, self.logger)
            await self.send_websocket_error(websocket, operation_id, [graphql_error])
            return

        if validation_errors:
            await self.send_websocket_error(
                websocket, operation_id, validation_errors
            )
            return

        operation_type = get_operation_type(document, operation_name)

        if operation_type == OperationType.SUBSCRIPTION:
            await self.start_websocket_subscription(
                websocket,
                operation_id,
                operation_name,
                variables,
                document,
                data,
                client_context,
            )
        else:
            await self.start_websocket_operation(
                websocket,
                operation_id,
                data,
                client_context,
            )

    async def start_websocket_operation(
        self,
        websocket: WebSocket,
        operation_id: str,
        data: dict,
        client_context: ClientContext,
    ):
        try:
            result = self.execute_graphql_query(websocket, data)
            if isawaitable(result):
                result = await result

            if isinstance(result, tuple) and len(result) == 2:
                success, payload = result
                if not success:
                    await self.send_websocket_error(websocket, operation_id, payload)
                    return
                result = payload

            await self.send_websocket_next(websocket, operation_id, result)
            await websocket.send_json(
                {"id": operation_id, "type": self.GQL_COMPLETE}
            )
        except GraphQLError as error:
            await self.send_websocket_error(websocket, operation_id, [error])
        except Exception as error:
            graphql_error = GraphQLError(str(error), original_error=error)
            log_error(graphql_error, self.logger)
            await self.send_websocket_error(websocket, operation_id, [graphql_error])

    async def start_websocket_subscription(
        self,
        websocket: WebSocket,
        operation_id: str,
        operation_name: Any,
        variables: Any,
        document: Any,
        data: dict,
        client_context: ClientContext,
    ):
        try:
            context_value = self.get_context_for_request(websocket, data)
            if isawaitable(context_value):
                context_value = await context_value

            root_value = self.get_root_value(
                websocket,
                data,
            )
            if isawaitable(root_value):
                root_value = await root_value

            result = subscribe(
                self.schema,
                document,
                context_value=context_value,
                root_value=root_value,
                variable_values=variables,
                operation_name=operation_name,
            )
            if isawaitable(result):
                result = await result

            if isinstance(result, ExecutionResult):
                await self.send_websocket_next(websocket, operation_id, result)
                await websocket.send_json(
                    {"id": operation_id, "type": self.GQL_COMPLETE}
                )
                return

            operation = Operation(
                operation_id,
                operation_name,
                cast(AsyncGenerator, result),
            )
            client_context.operations[operation_id] = operation
            client_context.operation_tasks[operation_id] = asyncio.create_task(
                self.observe_async_results(websocket, operation, client_context)
            )
        except GraphQLError as error:
            await self.send_websocket_error(websocket, operation_id, [error])
        except Exception as error:
            graphql_error = GraphQLError(str(error), original_error=error)
            log_error(graphql_error, self.logger)
            await self.send_websocket_error(websocket, operation_id, [graphql_error])

    async def observe_async_results(
        self,
        websocket: WebSocket,
        operation: Operation,
        client_context: ClientContext,
    ):
        completed = False
        try:
            async for result in operation.generator:
                await self.send_websocket_next(websocket, operation.id, result)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            graphql_error = (
                error
                if isinstance(error, GraphQLError)
                else GraphQLError(str(error), original_error=error)
            )
            log_error(graphql_error, self.logger)
            await self.send_websocket_error(websocket, operation.id, [graphql_error])
            completed = True
        finally:
            if client_context.operations.get(operation.id) is operation:
                client_context.operations.pop(operation.id, None)
                client_context.operation_tasks.pop(operation.id, None)

                if completed and WebSocketState.DISCONNECTED not in (
                    websocket.client_state,
                    websocket.application_state,
                ):
                    with suppress(WebSocketDisconnect):
                        await websocket.send_json(
                            {"id": operation.id, "type": self.GQL_COMPLETE}
                        )

    async def stop_websocket_operation(
        self,
        websocket: WebSocket,
        operation_id: str,
        client_context: ClientContext,
    ):
        operation = client_context.operations.pop(operation_id, None)
        task = client_context.operation_tasks.pop(operation_id, None)

        if task and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        if operation:
            generator = operation.generator
            aclose = getattr(generator, "aclose", None)
            if aclose:
                with suppress(Exception):
                    result = aclose()
                    if isawaitable(result):
                        await result

    async def send_websocket_next(
        self,
        websocket: WebSocket,
        operation_id: str,
        result: Any,
    ):
        if isinstance(result, ExecutionResult):
            payload = {"data": result.data}
            if result.errors:
                payload["errors"] = [
                    self.error_formatter(error, self.debug)
                    for error in result.errors
                ]
        else:
            payload = result

        await websocket.send_json(
            {
                "id": operation_id,
                "type": self.GQL_NEXT,
                "payload": payload,
            }
        )

    async def send_websocket_error(
        self,
        websocket: WebSocket,
        operation_id: str,
        errors: Any,
    ):
        if not isinstance(errors, (list, tuple)):
            errors = [errors]

        payload = [
            self.error_formatter(error, self.debug)
            if isinstance(error, GraphQLError)
            else error
            for error in errors
        ]

        await websocket.send_json(
            {
                "id": operation_id,
                "type": self.GQL_ERROR,
                "payload": payload,
            }
        )