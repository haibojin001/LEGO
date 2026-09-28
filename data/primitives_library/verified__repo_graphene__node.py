from functools import partial
from inspect import isclass

from ..types import Field, Interface, ObjectType
from ..types.interface import InterfaceOptions
from ..types.utils import get_type
from .id_type import BaseGlobalIDType, DefaultGlobalIDType


def is_node(objecttype):
    if not isclass(objecttype) or not issubclass(objecttype, ObjectType):
        return False
    return any(issubclass(interface, Node) for interface in objecttype._meta.interfaces)


class GlobalID(Field):
    def __init__(
        self,
        node=None,
        parent_type=None,
        required=True,
        global_id_type=DefaultGlobalIDType,
        *args,
        **kwargs,
    ):
        super().__init__(
            global_id_type.graphene_type,
            required=required,
            *args,
            **kwargs,
        )
        self.node = Node if node is None else node
        self.parent_type_name = (
            parent_type._meta.name if parent_type is not None else None
        )

    @staticmethod
    def id_resolver(parent_resolver, node, root, info, parent_type_name=None, **args):
        identifier = parent_resolver(root, info, **args)
        type_name = parent_type_name or info.parent_type.name
        return node.to_global_id(type_name, identifier)

    def wrap_resolve(self, parent_resolver):
        return partial(
            self.id_resolver,
            parent_resolver,
            self.node,
            parent_type_name=self.parent_type_name,
        )


class NodeField(Field):
    def __init__(self, node, type_=False, **kwargs):
        assert issubclass(node, Node), "NodeField can only operate in Nodes"
        self.node_type = node
        self.field_type = type_
        id_type = node._meta.global_id_type

        super().__init__(
            type_ or node,
            id=id_type.graphene_type(
                required=True,
                description="The ID of the object",
            ),
            **kwargs,
        )

    def wrap_resolve(self, parent_resolver):
        return partial(
            self.node_type.node_resolver,
            get_type(self.field_type),
        )


class AbstractNode(Interface):
    class Meta:
        abstract = True

    @classmethod
    def __init_subclass_with_meta__(
        cls,
        global_id_type=DefaultGlobalIDType,
        **options,
    ):
        assert issubclass(
            global_id_type,
            BaseGlobalIDType,
        ), "Custom ID type need to be implemented as a subclass of BaseGlobalIDType."

        meta = InterfaceOptions(cls)
        meta.global_id_type = global_id_type
        meta.fields = {
            "id": GlobalID(
                cls,
                global_id_type=global_id_type,
                description="The ID of the object",
            )
        }

        initializer = super(AbstractNode, cls).__init_subclass_with_meta__
        initializer(_meta=meta, **options)

    @classmethod
    def resolve_global_id(cls, info, global_id):
        return cls._meta.global_id_type.resolve_global_id(info, global_id)


class Node(AbstractNode):
    """An object with an ID"""

    @classmethod
    def Field(cls, *args, **kwargs):
        return NodeField(cls, *args, **kwargs)

    @classmethod
    def node_resolver(cls, only_type, root, info, id):
        return cls.get_node_from_global_id(info, id, only_type=only_type)

    @classmethod
    def get_node_from_global_id(cls, info, global_id, only_type=None):
        type_name, identifier = cls.resolve_global_id(info, global_id)

        schema_type = info.schema.get_type(type_name)
        if schema_type is None:
            raise Exception(f'Relay Node "{type_name}" not found in schema')

        graphene_type = schema_type.graphene_type

        if only_type:
            assert graphene_type == only_type, (
                f"Must receive a {only_type._meta.name} id."
            )

        if cls not in graphene_type._meta.interfaces:
            raise Exception(
                f'ObjectType "{type_name}" does not implement the "{cls}" interface.'
            )

        resolver = getattr(graphene_type, "get_node", None)
        if resolver:
            return resolver(info, identifier)

    @classmethod
    def to_global_id(cls, type_, id):
        return cls._meta.global_id_type.to_global_id(type_, id)