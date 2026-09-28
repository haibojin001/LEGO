from typing import cast

from graphql.type import GraphQLNamedType, GraphQLSchema, GraphQLUnionType

from .types import Resolver, SchemaBindable


class UnionType(SchemaBindable):
    """Bindable used to attach Python type resolution logic to GraphQL unions."""

    _resolve_type: Resolver | None

    def __init__(self, name: str, type_resolver: Resolver | None = None) -> None:
        self.name = name
        self._resolve_type = type_resolver

    def set_type_resolver(self, type_resolver: Resolver) -> Resolver:
        self._resolve_type = type_resolver
        return type_resolver

    type_resolver = set_type_resolver

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        graphql_type = schema.type_map.get(self.name)
        self.validate_graphql_type(graphql_type)
        graphql_union = cast(GraphQLUnionType, graphql_type)
        graphql_union.resolve_type = self._resolve_type

    def validate_graphql_type(self, graphql_type: GraphQLNamedType | None) -> None:
        if not graphql_type:
            raise ValueError(f"Type {self.name} is not defined in the schema")

        if not isinstance(graphql_type, GraphQLUnionType):
            raise ValueError(
                f"{self.name} is defined in the schema, "
                f"but it is instance of {type(graphql_type).__name__} "
                f"(expected {GraphQLUnionType.__name__})"
            )