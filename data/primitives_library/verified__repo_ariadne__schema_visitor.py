from collections.abc import Callable, Mapping
from types import FunctionType
from typing import Any, Literal, Protocol, TypeVar, cast

from graphql import is_named_type, value_from_ast_untyped
from graphql.execution.values import get_argument_values
from graphql.language import DirectiveLocation
from graphql.type import (
    GraphQLArgument,
    GraphQLDirective,
    GraphQLEnumType,
    GraphQLEnumValue,
    GraphQLField,
    GraphQLInputField,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNamedType,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLSchema,
    GraphQLUnionType,
)

VisitableSchemaType = (
    GraphQLSchema
    | GraphQLObjectType
    | GraphQLInterfaceType
    | GraphQLInputObjectType
    | GraphQLNamedType
    | GraphQLScalarType
    | GraphQLField
    | GraphQLArgument
    | GraphQLUnionType
    | GraphQLEnumType
    | GraphQLEnumValue
)

V = TypeVar("V", bound=VisitableSchemaType)
VisitableMap = dict[str, V]
IndexedObject = VisitableMap | tuple[V, ...]

Callback = Callable[..., Any]


def each(tuple_or_dict: IndexedObject, callback: Callback):
    if isinstance(tuple_or_dict, tuple):
        for value in tuple_or_dict:
            callback(value)
    else:
        for key, value in tuple_or_dict.items():
            callback(value, key)


def update_each_key(object_map: VisitableMap, callback: Callback):
    keys_to_remove: list[str] = []

    for key, value in object_map.copy().items():
        result = callback(value, key)
        if result is False:
            keys_to_remove.append(key)
        elif result is not None:
            object_map[key] = result

    for key in keys_to_remove:
        object_map.pop(key)


class SchemaVisitor(Protocol):
    @classmethod
    def implements_visitor_method(cls, method_name: str):
        if not method_name.startswith("visit_"):
            return False

        try:
            method = getattr(cls, method_name)
        except AttributeError:
            return False

        if not isinstance(method, FunctionType):
            return False

        if cls.__name__ == "SchemaVisitor":
            return True

        if method.__qualname__.startswith("SchemaVisitor"):
            return False

        return True

    def visit_schema(self, schema: GraphQLSchema) -> None:
        pass

    def visit_scalar(self, scalar: GraphQLScalarType) -> GraphQLScalarType:
        pass

    def visit_object(self, object_: GraphQLObjectType) -> GraphQLObjectType:
        pass

    def visit_field_definition(
        self,
        field: GraphQLField,
        object_type: GraphQLObjectType | GraphQLInterfaceType,
    ) -> GraphQLField:
        pass

    def visit_argument_definition(
        self,
        argument: GraphQLArgument,
        field: GraphQLField,
        object_type: GraphQLObjectType | GraphQLInterfaceType,
    ) -> GraphQLArgument:
        pass

    def visit_interface(self, interface: GraphQLInterfaceType) -> GraphQLInterfaceType:
        pass

    def visit_union(self, union: GraphQLUnionType) -> GraphQLUnionType:
        pass

    def visit_enum(self, type_: GraphQLEnumType) -> GraphQLEnumType:
        pass

    def visit_enum_value(
        self, value: GraphQLEnumValue, enum_type: GraphQLEnumType
    ) -> GraphQLEnumValue:
        pass

    def visit_input_object(
        self, object_: GraphQLInputObjectType
    ) -> GraphQLInputObjectType:
        pass

    def visit_input_field_definition(
        self, field: GraphQLInputField, object_type: GraphQLInputObjectType
    ) -> GraphQLInputField:
        pass


def visit_schema(
    schema: GraphQLSchema,
    visitor_selector: Callable[
        [VisitableSchemaType, str], list["SchemaDirectiveVisitor"]
    ],
) -> GraphQLSchema:
    def call_method(
        method_name: str, type_: VisitableSchemaType, *args: Any
    ) -> VisitableSchemaType | Literal[False]:
        for visitor in visitor_selector(type_, method_name):
            new_type = getattr(visitor, method_name)(type_, *args)

            if new_type is None:
                continue

            if method_name == "visit_schema" or isinstance(type_, GraphQLSchema):
                raise ValueError(
                    f"Method {method_name} cannot replace schema with {new_type}"
                )

            if new_type is False:
                return False

            type_ = new_type

        return type_

    def visit(
        type_: VisitableSchemaType,
    ) -> VisitableSchemaType | Literal[False]:
        if isinstance(type_, GraphQLSchema):
            call_method("visit_schema", type_)

            def start_visit(named_type, type_name):
                if not type_name.startswith("__"):
                    visit(named_type)

            update_each_key(type_.type_map, start_visit)
            return type_

        if isinstance(type_, GraphQLObjectType):
            new_object = cast(GraphQLObjectType, call_method("visit_object", type_))
            if new_object:
                visit_fields(new_object)
            return new_object

        if isinstance(type_, GraphQLInterfaceType):
            new_interface = cast(
                GraphQLInterfaceType, call_method("visit_interface", type_)
            )
            if new_interface:
                visit_fields(new_interface)
            return new_interface

        if isinstance(type_, GraphQLInputObjectType):
            new_input_object = cast(
                GraphQLInputObjectType, call_method("visit_input_object", type_)
            )
            if new_input_object:
                update_each_key(
                    new_input_object.fields,
                    lambda field, name: call_method(
                        "visit_input_field_definition", field, new_input_object
                    ),
                )
            return new_input_object

        if isinstance(type_, GraphQLScalarType):
            return call_method("visit_scalar", type_)

        if isinstance(type_, GraphQLUnionType):
            return call_method("visit_union", type_)

        if isinstance(type_, GraphQLEnumType):
            new_enum = cast(GraphQLEnumType, call_method("visit_enum", type_))
            if new_enum:
                update_each_key(
                    new_enum.values,
                    lambda value, name: call_method("visit_enum_value", value, name),
                )
            return new_enum

        raise ValueError(f"Unexpected schema type: {type_}")

    def visit_fields(type_: GraphQLObjectType | GraphQLInterfaceType):
        def update_field(field, _):
            new_field = cast(
                GraphQLField, call_method("visit_field_definition", field, type_)
            )
            if new_field:
                update_each_key(
                    new_field.args,
                    lambda argument, name: call_method(
                        "visit_argument_definition", argument, new_field, type_
                    ),
                )
            return new_field

        update_each_key(type_.fields, update_field)

    visit(schema)
    return schema


class SchemaDirectiveVisitor(SchemaVisitor):
    def __init__(
        self,
        name: str,
        args: dict[str, Any],
        visited_type: VisitableSchemaType,
        schema: GraphQLSchema,
        context: Any,
    ):
        self.name = name
        self.args = args
        self.visited_type = visited_type
        self.schema = schema
        self.context = context

    @classmethod
    def get_directive_declaration(
        cls, directive_name: str, schema: GraphQLSchema
    ) -> GraphQLDirective | None:
        return schema.get_directive(directive_name)

    @classmethod
    def visit_schema_directives(
        cls,
        schema: GraphQLSchema,
        directive_visitors: Mapping[str, type["SchemaDirectiveVisitor"]],
        context: Any = None,
    ) -> dict[str, list["SchemaDirectiveVisitor"]]:
        declared_directives = {
            name: visitor.get_directive_declaration(name, schema)
            for name, visitor in directive_visitors.items()
        }
        created_visitors: dict[str, list[SchemaDirectiveVisitor]] = {}

        locations: dict[str, DirectiveLocation] = {
            "visit_schema": DirectiveLocation.SCHEMA,
            "visit_scalar": DirectiveLocation.SCALAR,
            "visit_object": DirectiveLocation.OBJECT,
            "visit_field_definition": DirectiveLocation.FIELD_DEFINITION,
            "visit_argument_definition": DirectiveLocation.ARGUMENT_DEFINITION,
            "visit_interface": DirectiveLocation.INTERFACE,
            "visit_union": DirectiveLocation.UNION,
            "visit_enum": DirectiveLocation.ENUM,
            "visit_enum_value": DirectiveLocation.ENUM_VALUE,
            "visit_input_object": DirectiveLocation.INPUT_OBJECT,
            "visit_input_field_definition": DirectiveLocation.INPUT_FIELD_DEFINITION,
        }

        def directive_nodes_for(
            type_: VisitableSchemaType,
        ) -> list[Any]:
            nodes: list[Any] = []

            ast_node = getattr(type_, "ast_node", None)
            if ast_node is not None:
                nodes.extend(ast_node.directives or ())

            if isinstance(type_, GraphQLSchema) or is_named_type(type_):
                for extension in getattr(type_, "extension_ast_nodes", ()) or ():
                    nodes.extend(extension.directives or ())

            return nodes

        def select_visitors(
            type_: VisitableSchemaType, method_name: str
        ) -> list[SchemaDirectiveVisitor]:
            location = locations.get(method_name)
            if location is None:
                return []

            selected: list[SchemaDirectiveVisitor] = []

            for directive in directive_nodes_for(type_):
                directive_name = directive.name.value
                visitor_class = directive_visitors.get(directive_name)

                if (
                    visitor_class is None
                    or not visitor_class.implements_visitor_method(method_name)
                ):
                    continue

                declaration = declared_directives.get(directive_name)
                if declaration is not None:
                    if location not in declaration.locations:
                        continue
                    args = get_argument_values(declaration, directive)
                else:
                    args = {
                        argument.name.value: value_from_ast_untyped(argument.value)
                        for argument in directive.arguments or ()
                    }

                visitor = visitor_class(
                    directive_name,
                    args,
                    type_,
                    schema,
                    context,
                )
                created_visitors.setdefault(directive_name, []).append(visitor)
                selected.append(visitor)

            return selected

        visit_schema(schema, select_visitors)
        return created_visitors