from collections.abc import Callable
from enum import Enum
from typing import Any

from graphql import GraphQLEnumType, GraphQLInputField, GraphQLSchema

from .enums_values_visitor import (
    GraphQLASTEnumDefaultValueLocation,
    GraphQLASTEnumsValuesVisitor,
    GraphQLSchemaEnumDefaultValueLocation,
    GraphQLSchemaEnumsValuesVisitor,
)

__all__ = [
    "repair_schema_default_enum_values",
    "validate_schema_default_enum_values",
]


def validate_schema_default_enum_values(schema: GraphQLSchema) -> None:
    """Validate enum default values defined in a GraphQL schema.

    Raises ValueError when an input field or field argument has a default
    value containing an enum member that is not declared by its enum type.
    """
    GraphQLEnumsValuesValidatorVisitor(schema)


class GraphQLEnumsValuesValidatorVisitor(GraphQLASTEnumsValuesVisitor):
    def visit_ast_enum_default_value(
        self, location: GraphQLASTEnumDefaultValueLocation
    ):
        valid_values = self.enum_values[location.enum_name]

        if location.enum_value in valid_values:
            return

        if location.arg_name:
            raise ValueError(
                f"Undefined enum value '{location.enum_value}' for enum "
                f"'{location.enum_name}' in a default value of "
                f"'{location.arg_name}' argument for '{location.field_name}' "
                f"field on '{location.object_name}' type."
            )

        raise ValueError(
            f"Undefined enum value '{location.enum_value}' for enum "
            f"'{location.enum_name}' in a default value of "
            f"'{location.field_name}' field on '{location.object_name}' type."
        )


def repair_schema_default_enum_values(schema: GraphQLSchema) -> None:
    """Replace enum member-name defaults with their corresponding Python values."""
    _patch_enum_parse_value(schema)
    GraphQLSchemaEnumsValuesRepairVisitor(schema)


def _patch_enum_parse_value(schema: GraphQLSchema) -> None:
    def make_patched_parse_value(
        enum_type: GraphQLEnumType,
        original: Callable[[str], Any],
    ) -> Callable[[Any], Any]:
        def patched_parse_value(input_value: Any) -> Any:
            if input_value in enum_type._value_lookup:
                if isinstance(input_value, Enum) or not isinstance(input_value, str):
                    return input_value
            return original(input_value)

        return patched_parse_value

    for type_def in schema.type_map.values():
        if isinstance(type_def, GraphQLEnumType):
            type_def.parse_value = make_patched_parse_value(
                type_def,
                type_def.parse_value,
            )


class GraphQLSchemaEnumsValuesRepairVisitor(GraphQLSchemaEnumsValuesVisitor):
    def visit_schema_enum_default_value(
        self, location: GraphQLSchemaEnumDefaultValueLocation
    ):
        valid_values = self.enum_values[location.enum_name]
        valid_default = valid_values[location.enum_value]

        if location.default_value_path is not None:
            location.default_value[location.default_value_path] = valid_default
        elif location.arg_def:
            location.arg_def.default_value = valid_default
        elif isinstance(location.field_def, GraphQLInputField):
            location.field_def.default_value = valid_default