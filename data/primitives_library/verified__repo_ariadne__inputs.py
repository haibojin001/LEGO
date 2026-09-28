from typing import cast

from graphql import GraphQLInputObjectType, GraphQLSchema
from graphql.type.definition import GraphQLInputFieldOutType, GraphQLNamedType

from .types import SchemaBindable


class InputType(SchemaBindable):
    """Schema bindable used to configure GraphQL input object types."""

    _out_type: GraphQLInputFieldOutType | None
    _out_names: dict[str, str] | None

    def __init__(
        self,
        name: str,
        out_type: GraphQLInputFieldOutType | None = None,
        out_names: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self._out_type = out_type
        self._out_names = out_names

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        type_definition = schema.type_map.get(self.name)
        self.validate_graphql_type(type_definition)
        input_object = cast(GraphQLInputObjectType, type_definition)

        if self._out_type:
            input_object.out_type = self._out_type  # type: ignore

        if self._out_names:
            for graphql_field_name, python_field_name in self._out_names.items():
                if graphql_field_name not in input_object.fields:
                    raise ValueError(
                        f"Field {graphql_field_name} is not defined on type {self.name}"
                    )
                input_object.fields[graphql_field_name].out_name = python_field_name

    def validate_graphql_type(self, graphql_type: GraphQLNamedType | None) -> None:
        if not graphql_type:
            raise ValueError(f"Type {self.name} is not defined in the schema")

        if not isinstance(graphql_type, GraphQLInputObjectType):
            raise ValueError(
                f"{self.name} is defined in the schema, "
                f"but it is instance of {type(graphql_type).__name__} "
                f"(expected {GraphQLInputObjectType.__name__})"
            )