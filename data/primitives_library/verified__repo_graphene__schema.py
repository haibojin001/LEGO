from enum import Enum as PyEnum
import inspect
from functools import partial

from graphql import (
    default_type_resolver,
    get_introspection_query,
    graphql,
    graphql_sync,
    introspection_types,
    parse,
    print_schema,
    subscribe,
    validate,
    ExecutionResult,
    GraphQLArgument,
    GraphQLBoolean,
    GraphQLError,
    GraphQLEnumValue,
    GraphQLField,
    GraphQLFloat,
    GraphQLID,
    GraphQLInputField,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
)

from ..utils.str_converters import to_camel_case
from ..utils.get_unbound_function import get_unbound_function
from .definitions import (
    GrapheneEnumType,
    GrapheneGraphQLType,
    GrapheneInputObjectType,
    GrapheneInterfaceType,
    GrapheneObjectType,
    GrapheneScalarType,
    GrapheneUnionType,
)
from .dynamic import Dynamic
from .enum import Enum
from .field import Field
from .inputobjecttype import InputObjectType
from .interface import Interface
from .objecttype import ObjectType
from .resolver import get_default_resolver
from .scalars import ID, Boolean, Float, Int, Scalar, String
from .structures import List, NonNull
from .union import Union
from .utils import get_field_as


introspection_query = get_introspection_query()
IntrospectionSchema = introspection_types["__Schema"]


def assert_valid_root_type(type_):
    if type_ is None:
        return

    is_graphene_objecttype = inspect.isclass(type_) and issubclass(
        type_, ObjectType
    )
    is_graphql_objecttype = isinstance(type_, GraphQLObjectType)

    assert (
        is_graphene_objecttype or is_graphql_objecttype
    ), f"Type {type_} is not a valid ObjectType."


def is_graphene_type(type_):
    if isinstance(type_, (List, NonNull)):
        return True

    if inspect.isclass(type_) and issubclass(
        type_, (ObjectType, InputObjectType, Scalar, Interface, Union, Enum)
    ):
        return True


def is_type_of_from_possible_types(possible_types, root, _info):
    return isinstance(root, possible_types)


def identity_resolve(root, info, **arguments):
    return root


class TypeMap(dict):
    def __init__(
        self,
        query=None,
        mutation=None,
        subscription=None,
        types=None,
        auto_camelcase=True,
    ):
        assert_valid_root_type(query)
        assert_valid_root_type(mutation)
        assert_valid_root_type(subscription)

        if types is None:
            types = []

        for type_ in types:
            assert is_graphene_type(type_)

        self.auto_camelcase = auto_camelcase

        create_graphql_type = self.add_type

        self.query = create_graphql_type(query) if query else None
        self.mutation = create_graphql_type(mutation) if mutation else None
        self.subscription = (
            create_graphql_type(subscription) if subscription else None
        )
        self.types = [
            create_graphql_type(graphene_type) for graphene_type in types
        ]

    def add_type(self, graphene_type):
        if inspect.isfunction(graphene_type):
            graphene_type = graphene_type()

        if isinstance(graphene_type, List):
            return GraphQLList(self.add_type(graphene_type.of_type))

        if isinstance(graphene_type, NonNull):
            return GraphQLNonNull(self.add_type(graphene_type.of_type))

        try:
            name = graphene_type._meta.name
        except AttributeError:
            raise TypeError(
                f"Expected Graphene type, but received: {graphene_type}."
            )

        graphql_type = self.get(name)
        if graphql_type:
            return graphql_type

        if issubclass(graphene_type, ObjectType):
            graphql_type = self.create_objecttype(graphene_type)
        elif issubclass(graphene_type, InputObjectType):
            graphql_type = self.create_inputobjecttype(graphene_type)
        elif issubclass(graphene_type, Interface):
            graphql_type = self.create_interface(graphene_type)
        elif issubclass(graphene_type, Scalar):
            graphql_type = self.create_scalar(graphene_type)
        elif issubclass(graphene_type, Enum):
            graphql_type = self.create_enum(graphene_type)
        elif issubclass(graphene_type, Union):
            graphql_type = self.construct_union(graphene_type)
        else:
            raise TypeError(
                f"Expected Graphene type, but received: {graphene_type}."
            )

        self[name] = graphql_type
        return graphql_type

    @staticmethod
    def create_scalar(graphene_type):
        scalars = {
            String: GraphQLString,
            Int: GraphQLInt,
            Float: GraphQLFloat,
            Boolean: GraphQLBoolean,
            ID: GraphQLID,
        }

        if graphene_type in scalars:
            return scalars[graphene_type]

        return GrapheneScalarType(
            graphene_type=graphene_type,
            name=graphene_type._meta.name,
            description=graphene_type._meta.description,
            serialize=getattr(graphene_type, "serialize", None),
            parse_value=getattr(graphene_type, "parse_value", None),
            parse_literal=getattr(graphene_type, "parse_literal", None),
        )

    @staticmethod
    def create_enum(graphene_type):
        values = {}

        for name, value in graphene_type._meta.enum.__members__.items():
            description = getattr(value, "description", None)

            if isinstance(description, PyEnum):
                description = None

            if not description and callable(graphene_type._meta.description):
                description = graphene_type._meta.description(value)

            deprecation_reason = getattr(value, "deprecation_reason", None)

            if isinstance(deprecation_reason, PyEnum):
                deprecation_reason = None

            if not deprecation_reason and callable(
                graphene_type._meta.deprecation_reason
            ):
                deprecation_reason = graphene_type._meta.deprecation_reason(value)

            values[name] = GraphQLEnumValue(
                value=value,
                description=description,
                deprecation_reason=deprecation_reason,
            )

        description = graphene_type._meta.description
        if callable(description):
            description = description(None)

        return GrapheneEnumType(
            graphene_type=graphene_type,
            values=values,
            name=graphene_type._meta.name,
            description=description,
        )

    def create_objecttype(self, graphene_type):
        def interfaces():
            graphql_interfaces = []

            for graphene_interface in graphene_type._meta.interfaces:
                interface = self.add_type(graphene_interface)
                assert interface.graphene_type == graphene_interface
                graphql_interfaces.append(interface)

            return graphql_interfaces

        if graphene_type._meta.possible_types:
            is_type_of = partial(
                is_type_of_from_possible_types,
                graphene_type._meta.possible_types,
            )
        else:
            is_type_of = graphene_type.is_type_of

        return GrapheneObjectType(
            graphene_type=graphene_type,
            name=graphene_type._meta.name,
            description=graphene_type._meta.description,
            fields=partial(self.create_fields_for_type, graphene_type),
            is_type_of=is_type_of,
            interfaces=interfaces,
        )

    def create_interface(self, graphene_type):
        resolve_type = (
            partial(
                self.resolve_type,
                graphene_type.resolve_type,
                graphene_type._meta.name,
            )
            if graphene_type.resolve_type
            else None
        )

        def interfaces():
            graphql_interfaces = []

            for graphene_interface in graphene_type._meta.interfaces:
                interface = self.add_type(graphene_interface)
                assert interface.graphene_type == graphene_interface
                graphql_interfaces.append(interface)

            return graphql_interfaces

        return GrapheneInterfaceType(
            graphene_type=graphene_type,
            name=graphene_type._meta.name,
            description=graphene_type._meta.description,
            fields=partial(self.create_fields_for_type, graphene_type),
            interfaces=interfaces,
            resolve_type=resolve_type,
        )

    def create_inputobjecttype(self, graphene_type):
        return GrapheneInputObjectType(
            graphene_type=graphene_type,
            name=graphene_type._meta.name,
            description=graphene_type._meta.description,
            out_type=graphene_type._meta.container,
            fields=partial(
                self.create_fields_for_type,
                graphene_type,
                is_input_type=True,
            ),
        )

    def construct_union(self, graphene_type):
        resolve_type = (
            partial(
                self.resolve_type,
                graphene_type.resolve_type,
                graphene_type._meta.name,
            )
            if graphene_type.resolve_type
            else None
        )

        def types():
            graphql_types = []

            for graphene_objecttype in graphene_type._meta.types:
                objecttype = self.add_type(graphene_objecttype)
                assert objecttype.graphene_type == graphene_objecttype
                graphql_types.append(objecttype)

            return graphql_types

        return GrapheneUnionType(
            graphene_type=graphene_type,
            name=graphene_type._meta.name,
            description=graphene_type._meta.description,
            types=types,
            resolve_type=resolve_type,
        )

    def get_name(self, name):
        if self.auto_camelcase:
            return to_camel_case(name)
        return name

    def create_fields_for_type(self, graphene_type, is_input_type=False):
        fields = {}

        for name, field in graphene_type._meta.fields.items():
            if isinstance(field, Dynamic):
                field = get_field_as(field.get_type(self), _as=Field)

                if not field:
                    continue

            field_type = self.add_type(field.type)

            if field_type is None:
                continue

            field_name = field.name or self.get_name(name)

            if is_input_type:
                fields[field_name] = GraphQLInputField(
                    field_type,
                    default_value=field.default_value,
                    out_name=name,
                    description=field.description,
                    deprecation_reason=field.deprecation_reason,
                )
                continue

            arguments = {}

            for arg_name, arg in field.args.items():
                arg_type = self.add_type(arg.type)

                if arg_type is None:
                    continue

                processed_arg_name = arg.name or self.get_name(arg_name)

                arguments[processed_arg_name] = GraphQLArgument(
                    arg_type,
                    out_name=arg_name,
                    description=arg.description,
                    default_value=arg.default_value,
                    deprecation_reason=arg.deprecation_reason,
                )

            resolver = field.wrap_resolve(
                self.get_function_for_type(
                    graphene_type,
                    f"resolve_{name}",
                    name,
                    field.default_value,
                )
            )

            subscribe_resolver = None
            if field.wrap_subscribe:
                subscribe_resolver = field.wrap_subscribe(
                    self.get_function_for_type(
                        graphene_type,
                        f"subscribe_{name}",
                        name,
                        field.default_value,
                    )
                )

            fields[field_name] = GraphQLField(
                field_type,
                args=arguments,
                resolve=resolver,
                subscribe=subscribe_resolver,
                deprecation_reason=field.deprecation_reason,
                description=field.description,
            )

        return fields

    def get_function_for_type(
        self, graphene_type, func_name, name, default_value
    ):
        if graphene_type._meta.default_resolver:
            default_resolver = graphene_type._meta.default_resolver
        else:
            default_resolver = get_default_resolver()

        if func_name not in graphene_type.__dict__:
            for interface in graphene_type._meta.interfaces:
                if name in interface._meta.fields:
                    resolver = self.get_function_for_type(
                        interface,
                        func_name,
                        name,
                        default_value,
                    )
                    if resolver:
                        return resolver

            return partial(default_resolver, name, default_value)

        resolver = graphene_type.__dict__[func_name]

        if not resolver:
            return None

        return get_unbound_function(resolver)

    def resolve_type(self, resolve_type_func, type_name, root, info, _type):
        graphene_type = resolve_type_func(root, info)

        if inspect.isclass(graphene_type) and issubclass(
            graphene_type, ObjectType
        ):
            return graphene_type._meta.name

        return default_type_resolver(root, info, _type)


def normalize_execute_kwargs(kwargs):
    if "root" in kwargs:
        assert "root_value" not in kwargs, (
            "The `root` and `root_value` parameters cannot be used at the same time."
        )
        kwargs["root_value"] = kwargs.pop("root")

    if "context" in kwargs:
        assert "context_value" not in kwargs, (
            "The `context` and `context_value` parameters cannot be used at the same time."
        )
        kwargs["context_value"] = kwargs.pop("context")

    if "variables" in kwargs:
        assert "variable_values" not in kwargs, (
            "The `variables` and `variable_values` parameters cannot be used at the same time."
        )
        kwargs["variable_values"] = kwargs.pop("variables")

    if "operation" in kwargs:
        assert "operation_name" not in kwargs, (
            "The `operation` and `operation_name` parameters cannot be used at the same time."
        )
        kwargs["operation_name"] = kwargs.pop("operation")

    return kwargs


class Schema:
    def __init__(
        self,
        query=None,
        mutation=None,
        subscription=None,
        types=None,
        directives=None,
        auto_camelcase=True,
    ):
        self.query = query
        self.mutation = mutation
        self.subscription = subscription

        type_map = TypeMap(
            query,
            mutation,
            subscription,
            types,
            auto_camelcase=auto_camelcase,
        )

        self.graphql_schema = GraphQLSchema(
            query=type_map.query,
            mutation=type_map.mutation,
            subscription=type_map.subscription,
            types=type_map.types,
            directives=directives,
        )

    def __str__(self):
        return print_schema(self.graphql_schema)

    def execute(self, *args, **kwargs):
        kwargs = normalize_execute_kwargs(kwargs)
        return graphql_sync(self.graphql_schema, *args, **kwargs)

    async def execute_async(self, *args, **kwargs):
        kwargs = normalize_execute_kwargs(kwargs)
        return await graphql(self.graphql_schema, *args, **kwargs)

    async def subscribe(self, query, *args, **kwargs):
        kwargs = normalize_execute_kwargs(kwargs)

        try:
            document = parse(query)
        except GraphQLError as error:
            return ExecutionResult(data=None, errors=[error])

        validation_errors = validate(self.graphql_schema, document)

        if validation_errors:
            return ExecutionResult(data=None, errors=validation_errors)

        return await subscribe(self.graphql_schema, document, *args, **kwargs)

    def introspect(self):
        result = self.execute(introspection_query)
        assert not result.errors, (
            "The schema is not valid. Errors: {}".format(result.errors)
        )
        return result.data

    def get_type(self, type_name):
        return self.graphql_schema.get_type(type_name)

    def lazy(self, _type):
        return lambda: self.get_type(_type)