from typing import cast

from graphql.type import (
    GraphQLNamedType,
    GraphQLScalarLiteralParser,
    GraphQLScalarSerializer,
    GraphQLScalarType,
    GraphQLScalarValueParser,
    GraphQLSchema,
)

from .types import SchemaBindable


class ScalarType(SchemaBindable):
    """Bindable used to attach Python serialization logic to GraphQL scalars."""

    _serialize: GraphQLScalarSerializer | None
    _parse_value: GraphQLScalarValueParser | None
    _parse_literal: GraphQLScalarLiteralParser | None

    def __init__(
        self,
        name: str,
        *,
        serializer: GraphQLScalarSerializer | None = None,
        value_parser: GraphQLScalarValueParser | None = None,
        literal_parser: GraphQLScalarLiteralParser | None = None,
    ) -> None:
        self.name = name
        self._serialize = serializer
        self._parse_value = value_parser
        self._parse_literal = literal_parser

    def set_serializer(self, f: GraphQLScalarSerializer) -> GraphQLScalarSerializer:
        """Set function as serializer for this scalar."""
        self._serialize = f
        return f

    def set_value_parser(
        self, f: GraphQLScalarValueParser
    ) -> GraphQLScalarValueParser:
        """Set function as value parser for this scalar."""
        self._parse_value = f
        return f

    def set_literal_parser(
        self, f: GraphQLScalarLiteralParser
    ) -> GraphQLScalarLiteralParser:
        """Set function as literal parser for this scalar."""
        self._parse_literal = f
        return f

    serializer = set_serializer
    value_parser = set_value_parser
    literal_parser = set_literal_parser

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        """Bind this scalar's configured functions to a schema scalar."""
        graphql_type = schema.type_map.get(self.name)
        self.validate_graphql_type(graphql_type)
        graphql_type = cast(GraphQLScalarType, graphql_type)

        if self._serialize:
            graphql_type.serialize = self._serialize
        if self._parse_value:
            graphql_type.parse_value = self._parse_value
        if self._parse_literal:
            graphql_type.parse_literal = self._parse_literal

    def validate_graphql_type(
        self, graphql_type: GraphQLNamedType | None
    ) -> None:
        """Validate that schema type matching this bindable is a scalar."""
        if not graphql_type:
            raise ValueError(f"Scalar {self.name} is not defined in the schema")

        if not isinstance(graphql_type, GraphQLScalarType):
            raise ValueError(
                f"{self.name} is defined in the schema, but it is instance of "
                f"{type(graphql_type).__name__} (expected GraphQLScalarType)"
            )