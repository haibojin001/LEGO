from collections.abc import Callable
from typing import cast

from graphql.type import GraphQLNamedType, GraphQLObjectType, GraphQLSchema

from .resolvers import resolve_to
from .types import Resolver, SchemaBindable


class ObjectType(SchemaBindable):
    """A schema bindable that attaches Python resolvers to GraphQL object fields."""

    _resolvers: dict[str, Resolver]

    def __init__(self, name: str) -> None:
        self.name = name
        self._resolvers = {}

    def field(self, name: str) -> Callable[[Resolver], Resolver]:
        if not isinstance(name, str):
            raise ValueError(
                'field decorator should be passed a field name: @foo.field("name")'
            )
        return self.create_register_resolver(name)

    def create_register_resolver(self, name: str) -> Callable[[Resolver], Resolver]:
        def register_resolver(resolver: Resolver) -> Resolver:
            self._resolvers[name] = resolver
            return resolver

        return register_resolver

    def set_field(self, name, resolver: Resolver) -> Resolver:
        self._resolvers[name] = resolver
        return resolver

    def set_alias(self, name: str, to: str | Callable) -> None:
        if callable(to):
            self._resolvers[name] = to
        else:
            self._resolvers[name] = resolve_to(to)

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        graphql_type = schema.type_map.get(self.name)
        self.validate_graphql_type(graphql_type)
        self.bind_resolvers_to_graphql_type(
            cast(GraphQLObjectType, graphql_type)
        )

    def validate_graphql_type(self, graphql_type: GraphQLNamedType | None) -> None:
        if not graphql_type:
            raise ValueError(f"Type {self.name} is not defined in the schema")

        if not isinstance(graphql_type, GraphQLObjectType):
            raise ValueError(
                f"{self.name} is defined in the schema, "
                f"but it is instance of {type(graphql_type).__name__} "
                f"(expected {GraphQLObjectType.__name__})"
            )

    def bind_resolvers_to_graphql_type(self, graphql_type, replace_existing=True):
        for field_name, resolver in self._resolvers.items():
            if field_name not in graphql_type.fields:
                raise ValueError(
                    f"Field {field_name} is not defined on type {self.name}"
                )

            graphql_field = graphql_type.fields[field_name]
            if replace_existing or graphql_field.resolve is None:
                graphql_field.resolve = resolver


class QueryType(ObjectType):
    """Convenience ObjectType bound to the GraphQL Query type."""

    def __init__(self) -> None:
        super().__init__("Query")


class MutationType(ObjectType):
    """Convenience ObjectType bound to the GraphQL Mutation type."""

    def __init__(self) -> None:
        super().__init__("Mutation")