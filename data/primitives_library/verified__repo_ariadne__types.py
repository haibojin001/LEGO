from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Generator,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from graphql import (
    DocumentNode,
    ExecutionResult,
    GraphQLError,
    GraphQLResolveInfo,
    GraphQLSchema,
)
from graphql.utilities.type_info import TypeInfo
from graphql.validation.rules import ASTValidationRule
from starlette.websockets import WebSocket

__all__ = [
    "ExecutionResult",
    "Resolver",
    "GraphQLResult",
    "SubscriptionResult",
    "Subscriber",
    "ErrorFormatter",
    "ContextValue",
    "RootValue",
    "BaseProxyRootValue",
    "QueryParser",
    "QueryValidator",
    "ValidationRules",
    "ExtensionList",
    "Extensions",
    "Middleware",
    "MiddlewareList",
    "Middlewares",
    "Operation",
    "OnConnect",
    "OnDisconnect",
    "OnOperation",
    "OnComplete",
    "Extension",
    "SchemaBindable",
]

Resolver = Callable[..., Any]

GraphQLResult = tuple[bool, dict]

SubscriptionResult = tuple[
    bool, list[dict] | AsyncGenerator[ExecutionResult, None]
]

Subscriber = Callable[..., AsyncGenerator | Generator | Iterator]

ErrorFormatter = Callable[[GraphQLError, bool], dict]

ContextValue = Any | Callable[[Any, dict], Any]

RootValue = Any | Callable[
    [Any | None, str | None, dict | None, DocumentNode], Any
]


class BaseProxyRootValue:
    __slots__ = ("root_value",)

    root_value: dict | None

    def __init__(self, root_value: dict | None = None):
        self.root_value = root_value

    def update_result(self, result: GraphQLResult) -> GraphQLResult:
        return result


QueryParser = Callable[[ContextValue, dict[str, Any]], DocumentNode]


class QueryValidator(Protocol):
    def __call__(
        self,
        schema: GraphQLSchema,
        document_ast: DocumentNode,
        rules: Collection[type[ASTValidationRule]] | None = None,
        max_errors: int | None = None,
        type_info: TypeInfo | None = None,
    ) -> list[GraphQLError]:
        ...


ValidationRules = (
    Collection[type[ASTValidationRule]]
    | Callable[
        [ContextValue, DocumentNode, dict[str, Any]],
        Collection[type[ASTValidationRule]],
    ]
)


class Extension:
    def request_started(self, context: ContextValue) -> None:
        pass

    def request_finished(self, context: ContextValue) -> None:
        pass

    def resolve(
        self,
        next_: Resolver,
        obj: Any,
        info: GraphQLResolveInfo,
        **kwargs: Any,
    ) -> Any:
        return next_(obj, info, **kwargs)

    def has_errors(
        self, errors: list[GraphQLError], context: ContextValue
    ) -> None:
        pass

    def parse_started(self, context: ContextValue) -> None:
        pass

    def parse_finished(self, context: ContextValue) -> None:
        pass

    def validation_started(self, context: ContextValue) -> None:
        pass

    def validation_finished(self, context: ContextValue) -> None:
        pass

    def execution_started(self, context: ContextValue) -> None:
        pass

    def execution_finished(self, context: ContextValue) -> None:
        pass

    def format(self, context: ContextValue) -> dict | None:
        return None


ExtensionList = Sequence[Extension]

Extensions = ExtensionList | Callable[
    [ContextValue, DocumentNode, dict[str, Any]], ExtensionList
]

Middleware = Callable[..., Any]

MiddlewareList = Sequence[Middleware]

Middlewares = MiddlewareList | Callable[
    [ContextValue, DocumentNode, dict[str, Any]], MiddlewareList
]


@dataclass
class Operation:
    id: str
    name: str | None
    query: str
    variables: dict | None


OnConnect = Callable[[WebSocket, dict], Any]

OnDisconnect = Callable[[WebSocket], Any]

OnOperation = Callable[[WebSocket, Operation], Any]

OnComplete = Callable[[WebSocket, Operation], Any]


@runtime_checkable
class SchemaBindable(Protocol):
    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        ...