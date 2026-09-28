from collections.abc import Callable

from graphql import (
    GraphQLField,
    GraphQLInputObjectType,
    GraphQLObjectType,
    GraphQLSchema,
)

from .resolvers import resolve_to
from .utils import convert_camel_case_to_snake

SchemaNameConverter = Callable[[str, GraphQLSchema, tuple[str, ...]], str]

GRAPHQL_SPEC_TYPES = (
    "__Directive",
    "__EnumValue",
    "__Field",
    "__InputValue",
    "__Schema",
    "__Type",
)


def default_schema_name_converter(graphql_name: str, *_) -> str:
    return convert_camel_case_to_snake(graphql_name)


def convert_names_in_schema_args(
    name_converter: SchemaNameConverter,
    graphql_field: GraphQLField,
    schema: GraphQLSchema,
    object_name: str,
    field_name: str,
) -> None:
    for argument_name, argument in graphql_field.args.items():
        if argument.out_name:
            continue

        converted_name = name_converter(
            argument_name,
            schema,
            (object_name, field_name, argument_name),
        )
        if converted_name != argument_name:
            argument.out_name = converted_name


def convert_names_in_schema_object(
    name_converter: SchemaNameConverter,
    graphql_object: GraphQLObjectType,
    schema: GraphQLSchema,
) -> None:
    for field_name, field in graphql_object.fields.items():
        if field.args:
            convert_names_in_schema_args(
                name_converter,
                field,
                schema,
                graphql_object.name,
                field_name,
            )

        if field.resolve:
            continue

        converted_name = name_converter(
            field_name,
            schema,
            (graphql_object.name, field_name),
        )
        if converted_name != field_name:
            field.resolve = resolve_to(converted_name)


def convert_names_in_schema_input(
    name_converter: SchemaNameConverter,
    graphql_input: GraphQLInputObjectType,
    schema: GraphQLSchema,
) -> None:
    for field_name, field in graphql_input.fields.items():
        if field.out_name:
            continue

        converted_name = name_converter(
            field_name,
            schema,
            (graphql_input.name, field_name),
        )
        if converted_name != field_name:
            field.out_name = converted_name


def convert_schema_names(
    schema: GraphQLSchema,
    name_converter: SchemaNameConverter | None,
) -> None:
    converter = name_converter or default_schema_name_converter

    for type_name, graphql_type in schema.type_map.items():
        if (
            isinstance(graphql_type, GraphQLObjectType)
            and type_name not in GRAPHQL_SPEC_TYPES
        ):
            convert_names_in_schema_object(converter, graphql_type, schema)

        if isinstance(graphql_type, GraphQLInputObjectType):
            convert_names_in_schema_input(converter, graphql_type, schema)