import enum
from typing import Any, cast

from graphql.type import GraphQLEnumType, GraphQLNamedType, GraphQLSchema

from .types import SchemaBindable


class EnumType(SchemaBindable):
    """Connects GraphQL enum values to Python values."""

    def __init__(
        self,
        name: str,
        values: dict[str, Any] | type[enum.Enum] | type[enum.IntEnum],
    ) -> None:
        self.name = name
        self.values = cast(dict[str, Any], getattr(values, "__members__", values))

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        graphql_type = schema.type_map.get(self.name)
        self.validate_graphql_type(graphql_type)
        graphql_enum = cast(GraphQLEnumType, graphql_type)

        for name, value in self.values.items():
            if name not in graphql_enum.values:
                raise ValueError(f"Value {name} is not defined on enum {self.name}")
            graphql_enum.values[name].value = value

    def validate_graphql_type(
        self, graphql_type: GraphQLNamedType | None
    ) -> None:
        if not graphql_type:
            raise ValueError(f"Enum {self.name} is not defined in the schema")

        if not isinstance(graphql_type, GraphQLEnumType):
            raise ValueError(
                f"{self.name} is defined in the schema, but it is instance of "
                f"{type(graphql_type).__name__} (expected {GraphQLEnumType.__name__})"
            )