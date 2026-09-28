from __future__ import annotations

import json
from http import HTTPStatus
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, cast

from graphql import DocumentNode, MiddlewareManager
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.types import Receive, Scope, Send

from ...constants import DATA_TYPE_JSON, DATA_TYPE_MULTIPART
from ...exceptions import HttpBadRequestError, HttpError
from ...explorer import Explorer
from ...file_uploads import combine_multipart_data
from ...graphql import graphql
from ...types import (
    ContextValue,
    ExtensionList,
    Extensions,
    GraphQLResult,
    MiddlewareList,
    Middlewares,
)
from .base import GraphQLHttpHandlerBase

if TYPE_CHECKING:
    from ...subscription_handlers import SubscriptionHandler


class GraphQLHTTPHandler(GraphQLHttpHandlerBase):
    """Default ASGI handler for GraphQL HTTP requests."""

    def __init__(
        self,
        extensions: Extensions | None = None,
        middleware: Middlewares | None = None,
        middleware_manager_class: type[MiddlewareManager] | None = None,
        subscription_handlers: list[SubscriptionHandler] | None = None,
    ) -> None:
        super().__init__()
        self.extensions = extensions
        self.middleware = middleware
        self.middleware_manager_class = middleware_manager_class or MiddlewareManager
        self.subscription_handlers: list[SubscriptionHandler] = (
            subscription_handlers or []
        )

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope=scope, receive=receive)
        response = await self.handle_request(request)
        await response(scope, receive, send)

    async def handle_request_override(self, request: Request) -> Response | None:
        return None

    async def handle_request(self, request: Request) -> Response:
        response = await self.handle_request_override(request)
        if response is not None:
            return response

        if request.method == "GET":
            if self.execute_get_queries and request.query_params.get("query"):
                return await self.graphql_http_server(request)

            if self.introspection and self.explorer:
                return await self.render_explorer(request, self.explorer)

        if request.method == "POST":
            return await self.graphql_http_server(request)

        return self.handle_not_allowed_method(request)

    async def render_explorer(self, request: Request, explorer: Explorer) -> Response:
        explorer_html = explorer.html(request)
        if isawaitable(explorer_html):
            explorer_html = await explorer_html

        if explorer_html:
            return HTMLResponse(explorer_html)

        return self.handle_not_allowed_method(request)

    async def graphql_http_server(self, request: Request) -> Response:
        try:
            data = await self.extract_data_from_request(request)
        except HttpError as error:
            return PlainTextResponse(
                error.message or error.status,
                status_code=HTTPStatus.BAD_REQUEST,
            )

        if self.subscription_handlers and isinstance(data, dict) and self.schema:
            for handler in self.subscription_handlers:
                if handler.supports(request, data):
                    context_value = await self.get_context_for_request(request, data)
                    return await handler.handle(
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

        success, result = await self.execute_graphql_query(request, data)
        return await self.create_json_response(request, result, success)

    async def extract_data_from_request(self, request: Request) -> Any:
        if request.method == "GET":
            return {
                "query": request.query_params.get("query"),
                "operationName": request.query_params.get("operationName"),
                "variables": request.query_params.get("variables"),
            }

        content_type = request.headers.get("Content-Type", "")
        if content_type.startswith(DATA_TYPE_JSON):
            return await self.extract_data_from_json_request(request)

        if content_type.startswith(DATA_TYPE_MULTIPART):
            return await self.extract_data_from_multipart_request(request)

        raise HttpBadRequestError("Request body is not a valid JSON")

    async def extract_data_from_json_request(self, request: Request) -> Any:
        try:
            return await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HttpBadRequestError("Request body is not a valid JSON") from error

    async def extract_data_from_multipart_request(self, request: Request) -> Any:
        try:
            form = await request.form()
            operations = json.loads(form.get("operations") or "{}")
            files_map = json.loads(form.get("map") or "{}")
            return combine_multipart_data(
                operations,
                files_map,
                cast(dict[str, UploadFile], form),
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise HttpBadRequestError("Invalid multipart request") from error

    async def execute_graphql_query(
        self, request: Request, data: Any
    ) -> tuple[bool, GraphQLResult]:
        context_value = await self.get_context_for_request(request, data)
        extensions = await self.get_extensions_for_request(request, context_value)
        middleware = await self.get_middleware_for_request(request, context_value)

        return await graphql(
            self.schema,
            data,
            context_value=context_value,
            root_value=self.root_value,
            query_parser=self.query_parser,
            query_validator=self.query_validator,
            validation_rules=self.validation_rules,
            debug=self.debug,
            introspection=self.introspection,
            logger=self.logger,
            error_formatter=self.error_formatter,
            extensions=extensions,
            middleware=middleware,
            middleware_manager_class=self.middleware_manager_class,
        )

    async def get_context_for_request(
        self, request: Request, data: Any
    ) -> ContextValue:
        context_value = self.context_value

        if callable(context_value):
            context_value = context_value(request, data)
            if isawaitable(context_value):
                context_value = await context_value

        if context_value is None:
            context_value = {}

        if isinstance(context_value, dict):
            context_value["request"] = request

        return context_value

    async def get_extensions_for_request(
        self, request: Request, context: ContextValue
    ) -> ExtensionList | None:
        extensions = self.extensions

        if callable(extensions):
            extensions = extensions(request, context)
            if isawaitable(extensions):
                extensions = await extensions

        return extensions

    async def get_middleware_for_request(
        self, request: Request, context: ContextValue
    ) -> MiddlewareList | None:
        middleware = self.middleware

        if callable(middleware):
            middleware = middleware(request, context)
            if isawaitable(middleware):
                middleware = await middleware

        return middleware

    async def create_json_response(
        self,
        request: Request,
        result: GraphQLResult,
        success: bool,
    ) -> Response:
        return JSONResponse(
            result,
            status_code=HTTPStatus.OK if success else HTTPStatus.BAD_REQUEST,
        )