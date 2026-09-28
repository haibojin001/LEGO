from abc import ABC, abstractmethod
from inspect import isawaitable
from logging import Logger, LoggerAdapter
from typing import Any

from graphql import DocumentNode, ExecutionContext, GraphQLSchema, MiddlewareManager
from starlette.types import Receive, Scope, Send

from ...explorer import Explorer
from ...format_error import format_error
from ...types import (
    ContextValue,
    ErrorFormatter,
    GraphQLResult,
    OnComplete,
    OnConnect,
    OnDisconnect,
    OnOperation,
    QueryParser,
    QueryValidator,
    RootValue,
    ValidationRules,
)


class GraphQLHandlerBase(ABC):
    """Shared foundation for ASGI GraphQL handlers."""

    def __init__(self) -> None:
        self.schema: GraphQLSchema | None = None
        self.context_value: ContextValue | None = None
        self.debug: bool = False
        self.error_formatter: ErrorFormatter = format_error
        self.introspection: bool = True
        self.explorer: Explorer | None = None
        self.logger: None | str | Logger | LoggerAdapter = None
        self.root_value: RootValue | None = None
        self.query_parser: QueryParser | None = None
        self.query_validator: QueryValidator | None = None
        self.validation_rules: ValidationRules | None = None
        self.execute_get_queries: bool = False
        self.execution_context_class: type[ExecutionContext] | None = None
        self.middleware_manager_class: type[MiddlewareManager] | None = None

    @abstractmethod
    async def handle(self, scope: Scope, receive: Receive, send: Send):
        raise NotImplementedError(
            "Subclasses of GraphQLHandlerBase must implement the 'handle' method."
        )

    def configure(
        self,
        schema: GraphQLSchema,
        context_value: ContextValue | None = None,
        root_value: RootValue | None = None,
        query_parser: QueryParser | None = None,
        query_validator: QueryValidator | None = None,
        validation_rules: ValidationRules | None = None,
        execute_get_queries: bool = False,
        debug: bool = False,
        introspection: bool = True,
        explorer: Explorer | None = None,
        logger: None | str | Logger | LoggerAdapter = None,
        error_formatter: ErrorFormatter = format_error,
        execution_context_class: type[ExecutionContext] | None = None,
    ):
        self.context_value = context_value
        self.debug = debug
        self.error_formatter = error_formatter
        self.execute_get_queries = execute_get_queries
        self.execution_context_class = execution_context_class
        self.introspection = introspection
        self.explorer = explorer
        self.logger = logger
        self.query_parser = query_parser
        self.query_validator = query_validator
        self.root_value = root_value
        self.schema = schema
        self.validation_rules = validation_rules

    async def get_context_for_request(self, request: Any, data: Any) -> Any:
        if callable(self.context_value):
            result = self.context_value(request, data)
            if isawaitable(result):
                result = await result
            return result

        return self.context_value or {"request": request}


class GraphQLHttpHandlerBase(GraphQLHandlerBase):
    """Base class for handlers serving GraphQL over HTTP."""

    @abstractmethod
    async def handle_request(self, request: Any) -> Any:
        pass

    @abstractmethod
    async def execute_graphql_query(
        self,
        request: Any,
        data: Any,
        *,
        context_value: Any | None = None,
        query_document: DocumentNode | None = None,
    ) -> GraphQLResult:
        pass


class GraphQLWebsocketHandlerBase(GraphQLHandlerBase):
    """Base class for handlers serving GraphQL over WebSocket."""

    def __init__(
        self,
        on_connect: OnConnect | None = None,
        on_disconnect: OnDisconnect | None = None,
        on_operation: OnOperation | None = None,
        on_complete: OnComplete | None = None,
    ) -> None:
        super().__init__()
        self.http_handler: GraphQLHttpHandlerBase | None = None
        self.on_connect: OnConnect | None = on_connect
        self.on_disconnect: OnDisconnect | None = on_disconnect
        self.on_operation: OnOperation | None = on_operation
        self.on_complete: OnComplete | None = on_complete

    @abstractmethod
    async def handle_websocket(self, websocket: Any):
        pass

    def configure(
        self,
        *args,
        http_handler: GraphQLHttpHandlerBase | None = None,
        **kwargs,
    ):
        super().configure(*args, **kwargs)
        self.http_handler = http_handler