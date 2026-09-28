from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from graphql import (
    EnumValueNode,
    GraphQLArgument,
    GraphQLEnumType,
    GraphQLField,
    GraphQLInputField,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLType,
    InputValueDefinitionNode,
    ListValueNode,
    ObjectValueNode,
)


@dataclass
class GraphQLSchemaEnumDefaultValueLocation:
    enum_name: str
    enum_value: Any
    object_name: str
    object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType
    field_name: str
    field_def: GraphQLField | GraphQLInputField
    arg_name: str | None
    arg_def: GraphQLArgument | None
    default_value: Any
    default_value_path: str | int | None


@dataclass
class GraphQLASTEnumDefaultValueLocation:
    enum_name: str
    enum_value: Any
    object_name: str
    object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType
    field_name: str
    field_def: GraphQLField | GraphQLInputField
    arg_name: str | None
    arg_def: GraphQLArgument | None
    default_value: Any
    default_value_path: str | int | None


def unwrap_type(type_def: GraphQLType) -> GraphQLType:
    current_type = type_def
    while isinstance(current_type, GraphQLNonNull | GraphQLList):
        current_type = current_type.of_type
    return current_type


def unwrap_list_type(type_def: GraphQLType) -> GraphQLType:
    current_type = type_def
    while isinstance(current_type, GraphQLNonNull):
        current_type = current_type.of_type

    if isinstance(current_type, GraphQLList):
        return current_type.of_type

    return current_type


def is_graphql_list(type_def: GraphQLType) -> bool:
    current_type = type_def
    while isinstance(current_type, GraphQLNonNull):
        current_type = current_type.of_type
    return isinstance(current_type, GraphQLList)


def is_raw_enum_value(value: Any) -> bool:
    return isinstance(value, str) and not isinstance(value, Enum)


class GraphQLEnumsValuesVisitor:
    schema: GraphQLSchema
    enum_values: dict[str, dict[str, Any]]

    def __init__(self, schema: GraphQLSchema):
        self.enum_values = {}
        self.schema = schema
        self.visit_enum_types()
        self.visit_schema()

    def visit_enum_types(self) -> None:
        for type_def in self.schema.type_map.values():
            if isinstance(type_def, GraphQLEnumType):
                self.enum_values[type_def.name] = {
                    name: enum_value.value
                    for name, enum_value in type_def.values.items()
                }

    def visit_schema(self) -> None:
        raise NotImplementedError(
            "GraphQLEnumsValuesVisitor subclasses must implement 'visit_schema'"
        )


class GraphQLSchemaEnumsValuesVisitor(GraphQLEnumsValuesVisitor):
    def visit_schema(self) -> None:
        for type_def in self.schema.type_map.values():
            if type_def.name.startswith("__"):
                continue

            if isinstance(type_def, GraphQLObjectType | GraphQLInterfaceType):
                self.visit_object(type_def)

            if isinstance(type_def, GraphQLInputObjectType):
                self.visit_input(type_def)

    def visit_object(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType,
    ) -> None:
        for field_name, field_def in object_def.fields.items():
            for arg_name, arg_def in field_def.args.items():
                self.visit_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                )

    def visit_input(self, input_def: GraphQLInputObjectType) -> None:
        for field_name, field_def in input_def.fields.items():
            self.visit_value(input_def, field_name, field_def)

    def visit_value(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType,
        field_name: str,
        field_def: GraphQLField | GraphQLInputField,
        arg_name: str | None = None,
        arg_def: GraphQLArgument | None = None,
    ) -> None:
        source_def: GraphQLInputField | GraphQLArgument
        if isinstance(field_def, GraphQLInputField):
            source_def = field_def
        elif isinstance(arg_def, GraphQLArgument):
            source_def = arg_def
        else:
            return

        if not source_def.default_value:
            return

        if is_graphql_list(source_def.type) and isinstance(
            source_def.default_value, list
        ):
            self.visit_list_value(
                object_def,
                field_name,
                field_def,
                arg_name,
                arg_def,
                source_def.type,
                source_def.default_value,
            )
            return

        value_type = unwrap_type(source_def.type)
        if isinstance(value_type, GraphQLEnumType) and is_raw_enum_value(
            source_def.default_value
        ):
            self.visit_schema_enum_default_value(
                GraphQLSchemaEnumDefaultValueLocation(
                    enum_name=value_type.name,
                    enum_value=source_def.default_value,
                    object_name=object_def.name,
                    object_def=object_def,
                    field_name=field_name,
                    field_def=field_def,
                    arg_name=arg_name,
                    arg_def=arg_def,
                    default_value=source_def.default_value,
                    default_value_path=None,
                )
            )
        elif isinstance(value_type, GraphQLInputObjectType):
            self.visit_input_value(
                object_def,
                field_name,
                field_def,
                arg_name,
                arg_def,
                value_type,
                source_def.default_value,
            )

    def visit_list_value(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType,
        field_name: str,
        field_def: GraphQLField | GraphQLInputField,
        arg_name: str | None,
        arg_def: GraphQLArgument | None,
        value_def: GraphQLType,
        value: Any,
    ) -> None:
        item_type = unwrap_list_type(value_def)

        if is_graphql_list(item_type):
            for item in value:
                self.visit_list_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                    item_type,
                    item,
                )
        elif isinstance(item_type, GraphQLEnumType):
            for index, item in enumerate(value):
                if is_raw_enum_value(item):
                    self.visit_schema_enum_default_value(
                        GraphQLSchemaEnumDefaultValueLocation(
                            enum_name=item_type.name,
                            enum_value=item,
                            object_name=object_def.name,
                            object_def=object_def,
                            field_name=field_name,
                            field_def=field_def,
                            arg_name=arg_name,
                            arg_def=arg_def,
                            default_value=value,
                            default_value_path=index,
                        )
                    )
        elif isinstance(item_type, GraphQLInputObjectType):
            for item in value:
                if isinstance(item, dict):
                    self.visit_input_value(
                        object_def,
                        field_name,
                        field_def,
                        arg_name,
                        arg_def,
                        item_type,
                        item,
                    )

    def visit_input_value(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType,
        field_name: str,
        field_def: GraphQLField | GraphQLInputField,
        arg_name: str | None,
        arg_def: GraphQLArgument | None,
        value_def: GraphQLInputObjectType,
        value: dict,
    ) -> None:
        for input_name, input_def in value_def.fields.items():
            input_value = value.get(input_name)
            if input_value is None:
                continue

            input_type = unwrap_type(input_def.type)
            if is_graphql_list(input_def.type) and isinstance(input_value, list):
                self.visit_list_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                    input_def.type,
                    input_value,
                )
            elif isinstance(input_type, GraphQLEnumType) and is_raw_enum_value(
                input_value
            ):
                self.visit_schema_enum_default_value(
                    GraphQLSchemaEnumDefaultValueLocation(
                        enum_name=input_type.name,
                        enum_value=input_value,
                        object_name=object_def.name,
                        object_def=object_def,
                        field_name=field_name,
                        field_def=field_def,
                        arg_name=arg_name,
                        arg_def=arg_def,
                        default_value=value,
                        default_value_path=input_name,
                    )
                )
            elif isinstance(input_type, GraphQLInputObjectType) and isinstance(
                input_value, dict
            ):
                self.visit_input_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                    input_type,
                    input_value,
                )

    def visit_schema_enum_default_value(
        self,
        location: GraphQLSchemaEnumDefaultValueLocation,
    ) -> None:
        raise NotImplementedError(
            "GraphQLSchemaEnumsValuesVisitor subclasses must implement "
            "'visit_schema_enum_default_value'"
        )


class GraphQLASTEnumsValuesVisitor(GraphQLEnumsValuesVisitor):
    def visit_schema(self) -> None:
        for type_def in self.schema.type_map.values():
            if type_def.name.startswith("__"):
                continue

            if isinstance(type_def, GraphQLObjectType | GraphQLInterfaceType):
                self.visit_object(type_def)

            if isinstance(type_def, GraphQLInputObjectType):
                self.visit_input(type_def)

    def visit_object(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType,
    ) -> None:
        for field_name, field_def in object_def.fields.items():
            for arg_name, arg_def in field_def.args.items():
                self.visit_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                )

    def visit_input(self, input_def: GraphQLInputObjectType) -> None:
        for field_name, field_def in input_def.fields.items():
            self.visit_value(input_def, field_name, field_def)

    def visit_value(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType,
        field_name: str,
        field_def: GraphQLField | GraphQLInputField,
        arg_name: str | None = None,
        arg_def: GraphQLArgument | None = None,
    ) -> None:
        source_def: GraphQLInputField | GraphQLArgument
        if isinstance(field_def, GraphQLInputField):
            source_def = field_def
        elif isinstance(arg_def, GraphQLArgument):
            source_def = arg_def
        else:
            return

        ast_node = source_def.ast_node
        if not isinstance(ast_node, InputValueDefinitionNode):
            return

        default_value = cast(InputValueDefinitionNode, ast_node).default_value
        if default_value is None:
            return

        if is_graphql_list(source_def.type) and isinstance(
            default_value, ListValueNode
        ):
            self.visit_list_value(
                object_def,
                field_name,
                field_def,
                arg_name,
                arg_def,
                source_def.type,
                default_value,
            )
            return

        value_type = unwrap_type(source_def.type)
        if isinstance(value_type, GraphQLEnumType) and isinstance(
            default_value, EnumValueNode
        ):
            self.visit_ast_enum_default_value(
                GraphQLASTEnumDefaultValueLocation(
                    enum_name=value_type.name,
                    enum_value=default_value.value,
                    object_name=object_def.name,
                    object_def=object_def,
                    field_name=field_name,
                    field_def=field_def,
                    arg_name=arg_name,
                    arg_def=arg_def,
                    default_value=default_value,
                    default_value_path=None,
                )
            )
        elif isinstance(value_type, GraphQLInputObjectType) and isinstance(
            default_value, ObjectValueNode
        ):
            self.visit_input_value(
                object_def,
                field_name,
                field_def,
                arg_name,
                arg_def,
                value_type,
                default_value,
            )

    def visit_list_value(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType,
        field_name: str,
        field_def: GraphQLField | GraphQLInputField,
        arg_name: str | None,
        arg_def: GraphQLArgument | None,
        value_def: GraphQLType,
        value: ListValueNode,
    ) -> None:
        item_type = unwrap_list_type(value_def)

        if is_graphql_list(item_type):
            for item in value.values:
                if isinstance(item, ListValueNode):
                    self.visit_list_value(
                        object_def,
                        field_name,
                        field_def,
                        arg_name,
                        arg_def,
                        item_type,
                        item,
                    )
        elif isinstance(item_type, GraphQLEnumType):
            for index, item in enumerate(value.values):
                if isinstance(item, EnumValueNode):
                    self.visit_ast_enum_default_value(
                        GraphQLASTEnumDefaultValueLocation(
                            enum_name=item_type.name,
                            enum_value=item.value,
                            object_name=object_def.name,
                            object_def=object_def,
                            field_name=field_name,
                            field_def=field_def,
                            arg_name=arg_name,
                            arg_def=arg_def,
                            default_value=value,
                            default_value_path=index,
                        )
                    )
        elif isinstance(item_type, GraphQLInputObjectType):
            for item in value.values:
                if isinstance(item, ObjectValueNode):
                    self.visit_input_value(
                        object_def,
                        field_name,
                        field_def,
                        arg_name,
                        arg_def,
                        item_type,
                        item,
                    )

    def visit_input_value(
        self,
        object_def: GraphQLObjectType | GraphQLInterfaceType | GraphQLInputObjectType,
        field_name: str,
        field_def: GraphQLField | GraphQLInputField,
        arg_name: str | None,
        arg_def: GraphQLArgument | None,
        value_def: GraphQLInputObjectType,
        value: ObjectValueNode,
    ) -> None:
        object_fields = {
            object_field.name.value: object_field for object_field in value.fields
        }

        for input_name, input_def in value_def.fields.items():
            object_field = object_fields.get(input_name)
            if object_field is None:
                continue

            input_value = object_field.value
            input_type = unwrap_type(input_def.type)

            if is_graphql_list(input_def.type) and isinstance(
                input_value, ListValueNode
            ):
                self.visit_list_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                    input_def.type,
                    input_value,
                )
            elif isinstance(input_type, GraphQLEnumType) and isinstance(
                input_value, EnumValueNode
            ):
                self.visit_ast_enum_default_value(
                    GraphQLASTEnumDefaultValueLocation(
                        enum_name=input_type.name,
                        enum_value=input_value.value,
                        object_name=object_def.name,
                        object_def=object_def,
                        field_name=field_name,
                        field_def=field_def,
                        arg_name=arg_name,
                        arg_def=arg_def,
                        default_value=value,
                        default_value_path=input_name,
                    )
                )
            elif isinstance(input_type, GraphQLInputObjectType) and isinstance(
                input_value, ObjectValueNode
            ):
                self.visit_input_value(
                    object_def,
                    field_name,
                    field_def,
                    arg_name,
                    arg_def,
                    input_type,
                    input_value,
                )

    def visit_ast_enum_default_value(
        self,
        location: GraphQLASTEnumDefaultValueLocation,
    ) -> None:
        raise NotImplementedError(
            "GraphQLASTEnumsValuesVisitor subclasses must implement "
            "'visit_ast_enum_default_value'"
        )