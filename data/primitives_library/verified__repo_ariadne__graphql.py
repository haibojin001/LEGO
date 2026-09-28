from asyncio import ensure_future
from collections.abc import AsyncGenerator, Awaitable, Collection, Sequence
from inspect import isawaitable
from logging import Logger, LoggerAdapter
from typing import Any, cast

from graphql import (
    DocumentNode,
    ExecutionContext,
    ExecutionResult,
    GraphQLError,
    GraphQLSchema,
    MiddlewareManager,
    OperationDefinitionNode,
    TypeInfo,
    execute,
    execute_sync,
    parse,
)
from graphql import subscribe as _subscribe
from graphql.validation import specified_rules, validate
from graphql.validation.rules import ASTValidationRule

from .extensions import ExtensionManager
from .format_error import format_error
from .logger import log_error
from .types import (
    BaseProxyRootValue,
    ErrorFormatter,
    ExtensionList,
    GraphQLResult,
    MiddlewareList,
    QueryParser,
    QueryValidator,
    RootValue,
    SubscriptionResult,
    ValidationRules,
)
from .validation.introspection_disabled import IntrospectionDisabledRule


def validate_data(data: Any) -> None:
    if not isinstance(data, dict):
        raise GraphQLError("Invalid data provided.")

    if not isinstance(data.get("query"), str):
        raise GraphQLError("The query must be a string.")

    if "variables" in data and data["variables"] is not None:
        if not isinstance(data["variables"], dict):
            raise GraphQLError("Variables must be a dictionary.")


def parse_query(
    context_value: Any, query_parser: QueryParser | None, data: dict[str, Any]
) -> DocumentNode:
    if query_parser:
        return query_parser(context_value, data)

    return parse(data["query"])


def validate_query(
    schema: GraphQLSchema,
    document: DocumentNode,
    validation_rules: Collection[type[ASTValidationRule]] | None = None,
    *,
    enable_introspection: bool = True,
    query_validator: QueryValidator | None = None,
) -> list[GraphQLError]:
    rules: Sequence[type[ASTValidationRule]] = specified_rules

    if validation_rules:
        rules = tuple(rules) + tuple(validation_rules)

    if not enable_introspection:
        rules = tuple(rules) + (IntrospectionDisabledRule,)

    validator = query_validator or validate
    return validator(
        schema,
        document,
        rules=rules,
        max_errors=50,
        type_info=TypeInfo(schema),
    )


def _get_operation(
    document: DocumentNode, operation_name: str | None
) -> OperationDefinitionNode | None:
    operations = [
        definition
        for definition in document.definitions
        if isinstance(definition, OperationDefinitionNode)
    ]

    if operation_name:
        for operation in operations:
            if operation.name and operation.name.value == operation_name:
                return operation
        return None

    if len(operations) == 1:
        return operations[0]

    return None


def validate_operation_is_query(
    document: DocumentNode, operation_name: str | None
) -> None:
    operation = _get_operation(document, operation_name)

    if operation and operation.operation.value != "query":
        raise GraphQLError("Operation is not a query.")


def validate_operation_is_not_subscription(
    document: DocumentNode, operation_name: str | None
) -> None:
    operation = _get_operation(document, operation_name)

    if operation and operation.operation.value == "subscription":
        raise GraphQLError("Subscription operations are not supported.")


def handle_graphql_errors(
    errors: list[GraphQLError],
    *,
    logger: None | str | Logger | LoggerAdapter,
    error_formatter: ErrorFormatter,
    debug: bool,
    extension_manager: ExtensionManager,
) -> GraphQLResult:
    for error in errors:
        log_error(logger, error)

    result: dict[str, Any] = {
        "errors": [error_formatter(error, debug) for error in errors],
    }

    extensions = extension_manager.format()
    if extensions:
        result["extensions"] = extensions

    return False, result


def handle_query_result(
    result: ExecutionResult,
    *,
    logger: None | str | Logger | LoggerAdapter,
    error_formatter: ErrorFormatter,
    debug: bool,
    extension_manager: ExtensionManager,
) -> GraphQLResult:
    response: dict[str, Any] = {"data": result.data}
    success = True

    if result.errors:
        success = False
        response["errors"] = [
            error_formatter(error, debug) for error in result.errors
        ]

        for error in result.errors:
            log_error(logger, error)

    extensions = extension_manager.format()
    if extensions:
        response["extensions"] = extensions

    return success, response


async def graphql(
    schema: GraphQLSchema,
    data: Any,
    *,
    context_value: Any | None = None,
    root_value: RootValue | None = None,
    query_parser: QueryParser | None = None,
    query_validator: QueryValidator | None = None,
    query_document: DocumentNode | None = None,
    debug: bool = False,
    introspection: bool = True,
    logger: None | str | Logger | LoggerAdapter = None,
    validation_rules: ValidationRules | None = None,
    require_query: bool = False,
    error_formatter: ErrorFormatter = format_error,
    middleware: MiddlewareList = None,
    middleware_manager_class: type[MiddlewareManager] | None = None,
    extensions: ExtensionList | None = None,
    execution_context_class: type[ExecutionContext] | None = None,
    **kwargs: Any,
) -> GraphQLResult:
    result_update: BaseProxyRootValue | None = None
    extension_manager = ExtensionManager(extensions, context_value)

    with extension_manager.request():
        try:
            validate_data(data)
            variables = data.get("variables")
            operation_name = data.get("operationName")

            if query_document:
                document = query_document
            else:
                document = parse_query(context_value, query_parser, data)

            if callable(validation_rules):
                validation_rules = cast(
                    Collection[type[ASTValidationRule]] | None,
                    validation_rules(context_value, document, data),
                )

            validation_errors = validate_query(
                schema,
                document,
                validation_rules,
                enable_introspection=introspection,
                query_validator=query_validator,
            )
            if validation_errors:
                return handle_graphql_errors(
                    validation_errors,
                    logger=logger,
                    error_formatter=error_formatter,
                    debug=debug,
                    extension_manager=extension_manager,
                )

            if require_query:
                validate_operation_is_query(document, operation_name)
            else:
                validate_operation_is_not_subscription(document, operation_name)

            if callable(root_value):
                root_value = root_value(
                    context_value, operation_name, variables, document
                )
                if isawaitable(root_value):
                    root_value = await root_value

            if isinstance(root_value, BaseProxyRootValue):
                result_update = root_value
                root_value = root_value.root_value

            execution_result = execute(
                schema,
                document,
                root_value=root_value,
                context_value=context_value,
                variable_values=variables,
                operation_name=operation_name,
                execution_context_class=execution_context_class,
                middleware=extension_manager.as_middleware_manager(
                    middleware, middleware_manager_class
                ),
                **kwargs,
            )

            if isawaitable(execution_result):
                execution_result = await execution_result
        except GraphQLError as error:
            error_result = handle_graphql_errors(
                [error],
                logger=logger,
                error_formatter=error_formatter,
                debug=debug,
                extension_manager=extension_manager,
            )

            if result_update:
                return result_update.update_result(error_result)

            return error_result

        result = handle_query_result(
            execution_result,
            logger=logger,
            error_formatter=error_formatter,
            debug=debug,
            extension_manager=extension_manager,
        )

        if result_update:
            return result_update.update_result(result)

        return result


def graphql_sync(
    schema: GraphQLSchema,
    data: Any,
    *,
    context_value: Any | None = None,
    root_value: RootValue | None = None,
    query_parser: QueryParser | None = None,
    query_validator: QueryValidator | None = None,
    query_document: DocumentNode | None = None,
    debug: bool = False,
    introspection: bool = True,
    logger: None | str | Logger | LoggerAdapter = None,
    validation_rules: ValidationRules | None = None,
    require_query: bool = False,
    error_formatter: ErrorFormatter = format_error,
    middleware: MiddlewareList = None,
    middleware_manager_class: type[MiddlewareManager] | None = None,
    extensions: ExtensionList | None = None,
    execution_context_class: type[ExecutionContext] | None = None,
    **kwargs: Any,
) -> GraphQLResult:
    result_update: BaseProxyRootValue | None = None
    extension_manager = ExtensionManager(extensions, context_value)

    with extension_manager.request():
        try:
            validate_data(data)
            variables = data.get("variables")
            operation_name = data.get("operationName")

            if query_document:
                document = query_document
            else:
                document = parse_query(context_value, query_parser, data)

            if callable(validation_rules):
                validation_rules = cast(
                    Collection[type[ASTValidationRule]] | None,
                    validation_rules(context_value, document, data),
                )

            validation_errors = validate_query(
                schema,
                document,
                validation_rules,
                enable_introspection=introspection,
                query_validator=query_validator,
            )
            if validation_errors:
                return handle_graphql_errors(
                    validation_errors,
                    logger=logger,
                    error_formatter=error_formatter,
                    debug=debug,
                    extension_manager=extension_manager,
                )

            if require_query:
                validate_operation_is_query(document, operation_name)
            else:
                validate_operation_is_not_subscription(document, operation_name)

            if callable(root_value):
                root_value = root_value(
                    context_value, operation_name, variables, document
                )

                if isawaitable(root_value):
                    ensure_future(cast(Awaitable[Any], root_value))
                    raise RuntimeError(
                        "Root value resolver can't be asynchronous in synchronous query executor."
                    )

            if isinstance(root_value, BaseProxyRootValue):
                result_update = root_value
                root_value = root_value.root_value

            execution_result = execute_sync(
                schema,
                document,
                root_value=root_value,
                context_value=context_value,
                variable_values=variables,
                operation_name=operation_name,
                execution_context_class=execution_context_class,
                middleware=extension_manager.as_middleware_manager(
                    middleware, middleware_manager_class
                ),
                **kwargs,
            )

            if isawaitable(execution_result):
                ensure_future(cast(Awaitable[Any], execution_result))
                raise RuntimeError(
                    "GraphQL execution failed to complete synchronously."
                )
        except GraphQLError as error:
            error_result = handle_graphql_errors(
                [error],
                logger=logger,
                error_formatter=error_formatter,
                debug=debug,
                extension_manager=extension_manager,
            )

            if result_update:
                return result_update.update_result(error_result)

            return error_result

        result = handle_query_result(
            execution_result,
            logger=logger,
            error_formatter=error_formatter,
            debug=debug,
            extension_manager=extension_manager,
        )

        if result_update:
            return result_update.update_result(result)

        return result


async def _format_subscription_results(
    results: AsyncGenerator[ExecutionResult, None],
    *,
    logger: None | str | Logger | LoggerAdapter,
    error_formatter: ErrorFormatter,
    debug: bool,
    extension_manager: ExtensionManager,
    result_update: BaseProxyRootValue | None = None,
) -> AsyncGenerator[GraphQLResult, None]:
    async for result in results:
        formatted_result = handle_query_result(
            result,
            logger=logger,
            error_formatter=error_formatter,
            debug=debug,
            extension_manager=extension_manager,
        )

        if result_update:
            formatted_result = result_update.update_result(formatted_result)

        yield formatted_result


async def subscribe(
    schema: GraphQLSchema,
    data: Any,
    *,
    context_value: Any | None = None,
    root_value: RootValue | None = None,
    query_parser: QueryParser | None = None,
    query_validator: QueryValidator | None = None,
    query_document: DocumentNode | None = None,
    debug: bool = False,
    introspection: bool = True,
    logger: None | str | Logger | LoggerAdapter = None,
    validation_rules: ValidationRules | None = None,
    error_formatter: ErrorFormatter = format_error,
    middleware: MiddlewareList = None,
    middleware_manager_class: type[MiddlewareManager] | None = None,
    extensions: ExtensionList | None = None,
    execution_context_class: type[ExecutionContext] | None = None,
    **kwargs: Any,
) -> SubscriptionResult:
    result_update: BaseProxyRootValue | None = None
    extension_manager = ExtensionManager(extensions, context_value)

    with extension_manager.request():
        try:
            validate_data(data)
            variables = data.get("variables")
            operation_name = data.get("operationName")

            if query_document:
                document = query_document
            else:
                document = parse_query(context_value, query_parser, data)

            if callable(validation_rules):
                validation_rules = cast(
                    Collection[type[ASTValidationRule]] | None,
                    validation_rules(context_value, document, data),
                )

            validation_errors = validate_query(
                schema,
                document,
                validation_rules,
                enable_introspection=introspection,
                query_validator=query_validator,
            )
            if validation_errors:
                return handle_graphql_errors(
                    validation_errors,
                    logger=logger,
                    error_formatter=error_formatter,
                    debug=debug,
                    extension_manager=extension_manager,
                )

            if callable(root_value):
                root_value = root_value(
                    context_value, operation_name, variables, document
                )

                if isawaitable(root_value):
                    root_value = await root_value

            if isinstance(root_value, BaseProxyRootValue):
                result_update = root_value
                root_value = root_value.root_value

            subscription_result = _subscribe(
                schema,
                document,
                root_value=root_value,
                context_value=context_value,
                variable_values=variables,
                operation_name=operation_name,
                execution_context_class=execution_context_class,
                middleware=extension_manager.as_middleware_manager(
                    middleware, middleware_manager_class
                ),
                **kwargs,
            )

            if isawaitable(subscription_result):
                subscription_result = await subscription_result
        except GraphQLError as error:
            error_result = handle_graphql_errors(
                [error],
                logger=logger,
                error_formatter=error_formatter,
                debug=debug,
                extension_manager=extension_manager,
            )

            if result_update:
                return result_update.update_result(error_result)

            return error_result

        if isinstance(subscription_result, ExecutionResult):
            result = handle_query_result(
                subscription_result,
                logger=logger,
                error_formatter=error_formatter,
                debug=debug,
                extension_manager=extension_manager,
            )

            if result_update:
                return result_update.update_result(result)

            return result

        return (
            True,
            _format_subscription_results(
                subscription_result,
                logger=logger,
                error_formatter=error_formatter,
                debug=debug,
                extension_manager=extension_manager,
                result_update=result_update,
            ),
        )