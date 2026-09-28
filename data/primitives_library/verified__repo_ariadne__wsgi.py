import json
from collections.abc import Callable
from inspect import isawaitable
from typing import Any, cast
from urllib.parse import parse_qsl

from graphql import (
    ExecutionContext,
    GraphQLError,
    GraphQLSchema,
    MiddlewareManager,
)

from .constants import (
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_TEXT_HTML,
    CONTENT_TYPE_TEXT_PLAIN,
    DATA_TYPE_JSON,
    DATA_TYPE_MULTIPART,
    HttpStatusResponse,
)
from .exceptions import HttpBadRequestError, HttpError
from .explorer import Explorer, ExplorerGraphiQL
from .file_uploads import combine_multipart_data
from .format_error import format_error
from .graphql import graphql_sync
from .types import (
    ContextValue,
    ErrorFormatter,
    ExtensionList,
    GraphQLResult,
    MiddlewareList,
    QueryParser,
    QueryValidator,
    RootValue,
    ValidationRules,
)

try:
    from python_multipart import parse_form  # type: ignore[import-untyped]
except ImportError:

    def parse_form(*_args, **_kwargs):
        raise NotImplementedError(
            "WSGI file uploads requires 'python-multipart' library."
        )


__all__ = ["FormData", "GraphQL", "GraphQLMiddleware"]

Extensions = Callable[[Any, ContextValue | None], ExtensionList] | ExtensionList
Middlewares = Callable[[Any, ContextValue | None], MiddlewareList] | MiddlewareList


class FormData:
    def __init__(self) -> None:
        self.fields: dict[str, str] = {}
        self.files: dict[str, Any] = {}

    def on_field(self, field: Any) -> None:
        name = field.field_name.decode("utf-8")
        self.fields[name] = field.value.decode("utf-8")

    def on_file(self, file: Any) -> None:
        name = file.field_name.decode("utf-8")
        self.files[name] = file


class GraphQL:
    """WSGI application implementing the GraphQL server."""

    def __init__(
        self,
        schema: GraphQLSchema,
        *,
        context_value: ContextValue | None = None,
        root_value: RootValue | None = None,
        query_parser: QueryParser | None = None,
        query_validator: QueryValidator | None = None,
        validation_rules: ValidationRules | None = None,
        debug: bool = False,
        introspection: bool = True,
        explorer: Explorer | None = None,
        logger: str | None = None,
        error_formatter: ErrorFormatter = format_error,
        execute_get_queries: bool = False,
        extensions: Extensions | None = None,
        middleware: Middlewares | None = None,
        middleware_manager_class: type[MiddlewareManager] | None = None,
        execution_context_class: type[ExecutionContext] | None = None,
    ) -> None:
        self.context_value = context_value
        self.root_value = root_value
        self.query_parser = query_parser
        self.query_validator = query_validator
        self.validation_rules = validation_rules
        self.debug = debug
        self.introspection = introspection
        self.explorer = explorer or ExplorerGraphiQL()
        self.logger = logger
        self.error_formatter = error_formatter
        self.execute_get_queries = execute_get_queries
        self.extensions = extensions
        self.middleware = middleware
        self.middleware_manager_class = middleware_manager_class or MiddlewareManager
        self.execution_context_class = execution_context_class
        self.schema = schema

    def __call__(self, environ: dict, start_response: Callable) -> list[bytes]:
        try:
            return self.handle_request(environ, start_response)
        except GraphQLError as error:
            return self.handle_graphql_error(error, start_response)
        except HttpError as error:
            return self.handle_http_error(error, start_response)

    def handle_graphql_error(
        self, error: GraphQLError, start_response: Callable
    ) -> list[bytes]:
        start_response(
            HttpStatusResponse.BAD_REQUEST.value,
            [("Content-Type", CONTENT_TYPE_JSON)],
        )
        return [
            json.dumps({"errors": [{"message": error.message}]}).encode("utf-8")
        ]

    def handle_http_error(
        self, error: HttpError, start_response: Callable
    ) -> list[bytes]:
        start_response(error.status, [("Content-Type", CONTENT_TYPE_TEXT_PLAIN)])
        body = error.message or error.status
        return [str(body).encode("utf-8")]

    def handle_request(self, environ: dict, start_response: Callable) -> list[bytes]:
        method = environ["REQUEST_METHOD"]
        if method == "GET":
            return self.handle_get(environ, start_response)
        if method == "POST":
            return self.handle_post(environ, start_response)
        return self.handle_not_allowed_method(environ, start_response)

    def handle_get(self, environ: dict, start_response: Callable) -> list[bytes]:
        query_params = parse_query_string(environ)

        if self.execute_get_queries and query_params.get("query"):
            return self.graphql_http_server(environ, start_response, query_params)

        if self.introspection and self.explorer:
            return self.render_explorer(environ, start_response)

        return self.handle_not_allowed_method(environ, start_response)

    def handle_post(self, environ: dict, start_response: Callable) -> list[bytes]:
        return self.graphql_http_server(environ, start_response)

    def handle_not_allowed_method(
        self, environ: dict, start_response: Callable
    ) -> list[bytes]:
        start_response(
            HttpStatusResponse.METHOD_NOT_ALLOWED.value,
            [
                ("Content-Type", CONTENT_TYPE_TEXT_PLAIN),
                ("Allow", "GET, POST"),
            ],
        )
        return [b"Method Not Allowed"]

    def render_explorer(self, environ: dict, start_response: Callable) -> list[bytes]:
        explorer_html = self.explorer.html(environ)

        if isawaitable(explorer_html):
            raise RuntimeError("WSGI application cannot render an async explorer.")

        if not explorer_html:
            return self.handle_not_allowed_method(environ, start_response)

        start_response(
            HttpStatusResponse.OK.value,
            [("Content-Type", CONTENT_TYPE_TEXT_HTML)],
        )
        return [cast(str, explorer_html).encode("utf-8")]

    def graphql_http_server(
        self,
        environ: dict,
        start_response: Callable,
        data: dict[str, Any] | None = None,
    ) -> list[bytes]:
        request_data = data if data is not None else self.get_request_data(environ)
        context_value = self.get_context_for_request(environ, request_data)
        result = self.execute_query(environ, request_data, context_value)
        return self.return_response_from_result(result, start_response)

    def get_request_data(self, environ: dict) -> dict[str, Any]:
        content_type = environ.get("CONTENT_TYPE", "")

        if content_type.startswith(DATA_TYPE_JSON):
            return self.get_json_request_data(environ)

        if content_type.startswith(DATA_TYPE_MULTIPART):
            return self.get_multipart_request_data(environ)

        raise HttpBadRequestError("Request body is not a valid JSON.")

    def get_json_request_data(self, environ: dict) -> dict[str, Any]:
        request_body = environ["wsgi.input"].read()

        try:
            data = json.loads(request_body)
        except (TypeError, ValueError) as error:
            raise HttpBadRequestError("Request body is not a valid JSON.") from error

        if not isinstance(data, dict):
            raise HttpBadRequestError("Request body is not a valid JSON.")

        return data

    def get_multipart_request_data(self, environ: dict) -> dict[str, Any]:
        form_data = FormData()
        headers = {
            "Content-Type": environ.get("CONTENT_TYPE", ""),
            "Content-Length": environ.get("CONTENT_LENGTH", "0"),
        }

        try:
            parse_form(
                headers,
                environ["wsgi.input"],
                form_data.on_field,
                form_data.on_file,
            )
            operations = json.loads(form_data.fields["operations"])
            files_map = json.loads(form_data.fields["map"])
        except (KeyError, TypeError, ValueError) as error:
            raise HttpBadRequestError(
                "Request body is not a valid multipart form."
            ) from error

        return cast(
            dict[str, Any],
            combine_multipart_data(operations, files_map, form_data.files),
        )

    def get_context_for_request(
        self, environ: dict, data: dict[str, Any]
    ) -> ContextValue:
        context_value = self.context_value

        if callable(context_value):
            context_value = context_value(environ, data)

        if isawaitable(context_value):
            raise RuntimeError("WSGI application cannot use an async context value.")

        if context_value is None:
            return {"request": environ}

        return cast(ContextValue, context_value)

    def get_extensions_for_request(
        self, environ: dict, context_value: ContextValue
    ) -> ExtensionList | None:
        extensions = self.extensions

        if callable(extensions):
            extensions = extensions(environ, context_value)

        if isawaitable(extensions):
            raise RuntimeError("WSGI application cannot use async extensions.")

        return cast(ExtensionList | None, extensions)

    def get_middleware_for_request(
        self, environ: dict, context_value: ContextValue
    ) -> MiddlewareList | None:
        middleware = self.middleware

        if callable(middleware):
            middleware = middleware(environ, context_value)

        if isawaitable(middleware):
            raise RuntimeError("WSGI application cannot use async middleware.")

        return cast(MiddlewareList | None, middleware)

    def execute_query(
        self,
        environ: dict,
        data: dict[str, Any],
        context_value: ContextValue,
    ) -> tuple[bool, GraphQLResult]:
        extensions = self.get_extensions_for_request(environ, context_value)
        middleware = self.get_middleware_for_request(environ, context_value)

        return graphql_sync(
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
            execution_context_class=self.execution_context_class,
        )

    def return_response_from_result(
        self, result: tuple[bool, GraphQLResult], start_response: Callable
    ) -> list[bytes]:
        success, response = result
        status = (
            HttpStatusResponse.OK.value
            if success
            else HttpStatusResponse.BAD_REQUEST.value
        )
        start_response(status, [("Content-Type", CONTENT_TYPE_JSON)])
        return [json.dumps(response).encode("utf-8")]


class GraphQLMiddleware:
    def __init__(
        self,
        app: Callable,
        schema: GraphQLSchema,
        path: str = "/graphql",
        **kwargs: Any,
    ) -> None:
        self.app = app
        self.path = path
        self.graphql = GraphQL(schema, **kwargs)

    def __call__(self, environ: dict, start_response: Callable) -> list[bytes]:
        if environ.get("PATH_INFO") == self.path:
            return self.graphql(environ, start_response)
        return self.app(environ, start_response)


def parse_query_string(environ: dict) -> dict[str, Any]:
    query_string = environ.get("QUERY_STRING", "")
    data: dict[str, Any] = dict(parse_qsl(query_string))

    if "variables" in data:
        try:
            data["variables"] = json.loads(data["variables"])
        except (TypeError, ValueError) as error:
            raise HttpBadRequestError("Variables are invalid JSON.") from error

    if "extensions" in data:
        try:
            data["extensions"] = json.loads(data["extensions"])
        except (TypeError, ValueError) as error:
            raise HttpBadRequestError("Extensions are invalid JSON.") from error

    return data