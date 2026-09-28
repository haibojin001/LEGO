from enum import Enum
from typing import cast

from graphql import GraphQLSchema, assert_valid_schema, build_ast_schema, parse

from .enums import EnumType
from .enums_default_values import (
    repair_schema_default_enum_values,
    validate_schema_default_enum_values,
)
from .schema_names import SchemaNameConverter, convert_schema_names
from .schema_visitor import SchemaDirectiveVisitor
from .types import SchemaBindable


SchemaBindables = SchemaBindable | type[Enum] | list[SchemaBindable | type[Enum]]


def _get_schema_bindable(value: SchemaBindable | type[Enum]) -> SchemaBindable:
    if isinstance(value, type) and issubclass(value, Enum):
        value = EnumType(value.__name__, value)
    return cast(SchemaBindable, value)


def make_executable_schema(
    type_defs: str | list[str],
    *bindables: SchemaBindables,
    directives: dict[str, type[SchemaDirectiveVisitor]] | None = None,
    convert_names_case: bool | SchemaNameConverter = False,
) -> GraphQLSchema:
    if isinstance(type_defs, list):
        type_defs = "\n".join(type_defs)

    schema = build_ast_schema(parse(type_defs))

    for bindable in bindables:
        items = bindable if isinstance(bindable, list) else (bindable,)
        for item in items:
            _get_schema_bindable(item).bind_to_schema(schema)

    if directives:
        SchemaDirectiveVisitor.visit_schema_directives(schema, directives)

    if convert_names_case:
        convert_schema_names(schema, convert_names_case)

    assert_valid_schema(schema)
    repair_schema_default_enum_values(schema)
    validate_schema_default_enum_values(schema)

    return schema