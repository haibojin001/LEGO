from contextlib import contextmanager

from graphql import GraphQLError
from graphql.execution import MiddlewareManager

from .types import ContextValue, ExtensionList, MiddlewareList


class ExtensionManager:
    __slots__ = ("context", "extensions", "extensions_reversed")

    def __init__(
        self,
        extensions: ExtensionList | None = None,
        context: ContextValue | None = None,
    ) -> None:
        self.context = context
        instances = tuple(factory() for factory in extensions) if extensions else ()
        self.extensions = instances
        self.extensions_reversed = instances[::-1]

    def as_middleware_manager(
        self,
        middleware: MiddlewareList = None,
        manager_class: type[MiddlewareManager] | None = None,
    ) -> MiddlewareManager | None:
        if not middleware and not self.extensions:
            return None

        manager_type = manager_class or MiddlewareManager
        return manager_type(*(middleware or []), *self.extensions)

    @contextmanager
    def request(self):
        for item in self.extensions:
            item.request_started(self.context)

        try:
            yield
        finally:
            for item in self.extensions_reversed:
                item.request_finished(self.context)

    def has_errors(self, errors: list[GraphQLError]):
        for item in self.extensions:
            item.has_errors(errors, self.context)

    def format(self) -> dict:
        output = {}
        for item in self.extensions:
            contribution = item.format(self.context)
            if contribution:
                output.update(contribution)
        return output