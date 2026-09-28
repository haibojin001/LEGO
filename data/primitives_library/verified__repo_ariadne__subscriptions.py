import inspect
from collections.abc import Callable
from typing import Any, cast

try:
    from anyio import to_thread
except ImportError:
    import asyncio

    class _ThreadAdapter:
        @staticmethod
        async def run_sync(func: Callable[..., Any], *args: Any) -> Any:
            return await asyncio.to_thread(func, *args)

    to_thread: _ThreadAdapter = _ThreadAdapter()

from graphql.type import GraphQLSchema

from .objects import ObjectType
from .types import Subscriber


class SubscriptionType(ObjectType):
    """Bindable populating the Subscription type in a GraphQL schema."""

    _subscribers: dict[str, Subscriber]

    def __init__(self) -> None:
        super().__init__("Subscription")
        self._subscribers = {}

    def source(self, name: str) -> Callable[[Subscriber], Subscriber]:
        """Return a decorator registering a subscription source for a field."""
        if not isinstance(name, str):
            raise ValueError(
                'source decorator should be passed a field name: @foo.source("name")'
            )
        return self.create_register_subscriber(name)

    def create_register_subscriber(
        self, name: str
    ) -> Callable[[Subscriber], Subscriber]:
        """Return a decorator registering a subscription source for a field."""

        def register_subscriber(generator: Subscriber) -> Subscriber:
            is_sync_generator = inspect.isgeneratorfunction(
                generator
            ) and not inspect.isasyncgenfunction(generator)

            if is_sync_generator:

                async def async_wrapper(*args: Any, **kwargs: Any):
                    sync_gen = generator(*args, **kwargs)

                    try:
                        while True:
                            try:

                                def get_next(gen: Any) -> Any:
                                    return next(gen)

                                value = await to_thread.run_sync(get_next, sync_gen)
                                yield value
                            except StopIteration:
                                break
                            except RuntimeError as error:
                                if isinstance(
                                    error.__cause__, StopIteration
                                ) or "StopIteration" in str(error):
                                    break
                                raise
                    finally:
                        try:
                            if hasattr(sync_gen, "close"):
                                close_fn = cast(Callable[[], None], sync_gen.close)
                                await to_thread.run_sync(close_fn)
                        except (Exception, GeneratorExit):
                            pass

                wrapped: Subscriber = async_wrapper
                self._subscribers[name] = wrapped
                return wrapped

            self._subscribers[name] = generator
            return generator

        return register_subscriber

    def set_source(self, name, generator: Subscriber) -> Subscriber:
        """Set a subscription source for a field."""
        return self.create_register_subscriber(name)(generator)

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        """Bind registered resolvers and subscribers to the schema."""
        super().bind_to_schema(schema)

        subscription_type = schema.type_map.get("Subscription")
        if subscription_type:
            for field_name, subscriber in self._subscribers.items():
                field = subscription_type.fields.get(field_name)
                if field:
                    field.subscribe = subscriber