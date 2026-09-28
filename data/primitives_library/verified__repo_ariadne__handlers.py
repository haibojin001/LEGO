from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from logging import Logger, LoggerAdapter
from typing import Any, cast

from graphql import DocumentNode, ExecutionResult, GraphQLError, GraphQLSchema
from starlette.requests import Request
from starlette.responses import Response

from ..graphql import subscribe, validate_data
from ..logger import log_error
from ..types import (
    ErrorFormatter,
    QueryParser,
    QueryValidator,
    RootValue,
    ValidationRules,
)
from .events import SubscriptionEvent, SubscriptionEventType


class SubscriptionHandler(ABC):
    @abstractmethod
    def supports(self, request: Request, data: dict) -> bool:
        ...

    @abstractmethod
    async def handle(
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
        ...

    async def generate_events(
        self,
        data: dict,
        *,
        schema: GraphQLSchema,
        context_value: Any,
        root_value: RootValue | None,
        query_parser: QueryParser | None,
        query_validator: QueryValidator | None,
        query_document: DocumentNode | None,
        validation_rules: ValidationRules | None,
        debug: bool,
        introspection: bool,
        logger: None | str | Logger | LoggerAdapter,
        error_formatter: ErrorFormatter,
    ) -> AsyncGenerator[SubscriptionEvent, None]:
        try:
            validate_data(data)

            successful, subscription = await subscribe(
                schema,
                data,
                context_value=context_value,
                root_value=root_value,
                query_parser=query_parser,
                query_document=query_document,
                query_validator=query_validator,
                validation_rules=validation_rules,
                debug=debug,
                introspection=introspection,
                logger=logger,
                error_formatter=error_formatter,
            )

            if not successful:
                if isinstance(subscription, list):
                    errors = cast(list[dict[str, Any]], subscription)
                else:
                    errors = cast(list[dict[str, Any]], [subscription])

                yield SubscriptionEvent(
                    event_type=SubscriptionEventType.ERROR,
                    result=ExecutionResult(
                        errors=[
                            GraphQLError(
                                str(item.get("message", ""))
                                if isinstance(item, Mapping)
                                else str(item)
                            )
                            for item in errors
                        ]
                    ),
                )
            else:
                results = cast(AsyncGenerator[ExecutionResult, None], subscription)

                try:
                    async for result in results:
                        yield SubscriptionEvent(
                            event_type=SubscriptionEventType.NEXT,
                            result=result,
                        )
                except (Exception, GraphQLError) as error:
                    if not isinstance(error, GraphQLError):
                        error = GraphQLError(str(error), original_error=error)
                        log_error(error, logger)

                    yield SubscriptionEvent(
                        event_type=SubscriptionEventType.ERROR,
                        result=ExecutionResult(errors=[error]),
                    )
        except GraphQLError as error:
            log_error(error, logger)
            yield SubscriptionEvent(
                event_type=SubscriptionEventType.ERROR,
                result=ExecutionResult(errors=[error]),
            )

        yield SubscriptionEvent(event_type=SubscriptionEventType.COMPLETE)