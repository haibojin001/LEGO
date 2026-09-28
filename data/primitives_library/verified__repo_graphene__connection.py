import re
from collections.abc import Iterable
from functools import partial
from typing import Type

from graphql_relay import connection_from_array

from ..types import Boolean, Enum, Int, Interface, List, NonNull, Scalar, String, Union
from ..types.field import Field
from ..types.objecttype import ObjectType, ObjectTypeOptions
from ..utils.thenables import maybe_thenable
from .node import AbstractNode, is_node


def get_edge_class(
    connection_class: Type["Connection"],
    _node: Type[AbstractNode],
    base_name: str,
    strict_types: bool = False,
):
    supplied_edge = getattr(connection_class, "Edge", None)
    edge_type_name = "{}Edge".format(base_name)

    class GeneratedEdgeBase:
        node = Field(
            NonNull(_node) if strict_types else _node,
            description="The item at the end of the edge",
        )
        cursor = String(required=True, description="A cursor for use in pagination")

    class EdgeMeta:
        description = "A Relay edge containing a `{}` and its cursor.".format(
            base_name
        )

    parents = [GeneratedEdgeBase]
    if supplied_edge:
        parents.insert(0, supplied_edge)
    if not isinstance(supplied_edge, ObjectType):
        parents.append(ObjectType)

    return type(edge_type_name, tuple(parents), {"Meta": EdgeMeta})


class PageInfo(ObjectType):
    class Meta:
        description = (
            "The Relay compliant `PageInfo` type, containing data necessary to"
            " paginate this connection."
        )

    has_next_page = Boolean(
        required=True,
        name="hasNextPage",
        description="When paginating forwards, are there more items?",
    )
    has_previous_page = Boolean(
        required=True,
        name="hasPreviousPage",
        description="When paginating backwards, are there more items?",
    )
    start_cursor = String(
        name="startCursor",
        description="When paginating backwards, the cursor to continue.",
    )
    end_cursor = String(
        name="endCursor",
        description="When paginating forwards, the cursor to continue.",
    )


def page_info_adapter(startCursor, endCursor, hasPreviousPage, hasNextPage):
    """Adapter for creating PageInfo instances"""
    return PageInfo(
        start_cursor=startCursor,
        end_cursor=endCursor,
        has_previous_page=hasPreviousPage,
        has_next_page=hasNextPage,
    )


class ConnectionOptions(ObjectTypeOptions):
    node = None


class Connection(ObjectType):
    class Meta:
        abstract = True

    @classmethod
    def __init_subclass_with_meta__(
        cls, node=None, name=None, strict_types=False, _meta=None, **options
    ):
        if _meta is None:
            _meta = ConnectionOptions(cls)

        assert node, "You have to provide a node in {}.Meta".format(cls.__name__)
        assert isinstance(node, NonNull) or issubclass(
            node, (Scalar, Enum, ObjectType, Interface, Union, NonNull)
        ), 'Received incompatible node "{}" for Connection {}.'.format(
            node, cls.__name__
        )

        raw_name = name or cls.__name__
        base_name = re.sub("Connection$", "", raw_name) or node._meta.name
        if name is None:
            name = "{}Connection".format(base_name)

        options["name"] = name
        _meta.node = node

        if not _meta.fields:
            _meta.fields = {}

        if "page_info" not in _meta.fields:
            _meta.fields["page_info"] = Field(
                PageInfo,
                name="pageInfo",
                required=True,
                description="Pagination data for this connection.",
            )

        if "edges" not in _meta.fields:
            edge_type = get_edge_class(cls, node, base_name, strict_types)
            cls.Edge = edge_type
            _meta.fields["edges"] = Field(
                NonNull(List(NonNull(edge_type) if strict_types else edge_type)),
                description="Contains the nodes in this connection.",
            )

        return super(Connection, cls).__init_subclass_with_meta__(
            _meta=_meta, **options
        )


def connection_adapter(cls, edges, pageInfo):
    """Adapter for creating Connection instances"""
    return cls(edges=edges, page_info=pageInfo)


class IterableConnectionField(Field):
    def __init__(self, type_, *args, **kwargs):
        kwargs.setdefault("before", String())
        kwargs.setdefault("after", String())
        kwargs.setdefault("first", Int())
        kwargs.setdefault("last", Int())
        super(IterableConnectionField, self).__init__(type_, *args, **kwargs)

    @property
    def type(self):
        declared_type = super(IterableConnectionField, self).type
        connection_type = (
            declared_type.of_type
            if isinstance(declared_type, NonNull)
            else declared_type
        )

        if is_node(connection_type):
            raise Exception(
                "ConnectionFields now need a explicit ConnectionType for Nodes.\n"
                "Read more: https://github.com/graphql-python/graphene/blob/v2.0.0/UPGRADE-v2.0.md#node-connections"
            )

        assert issubclass(connection_type, Connection), (
            '{} type has to be a subclass of Connection. Received "{}".'.format(
                self.__class__.__name__, connection_type
            )
        )
        return declared_type

    @classmethod
    def resolve_connection(cls, connection_type, args, resolved):
        if isinstance(resolved, connection_type):
            return resolved

        assert isinstance(resolved, Iterable), (
            "Resolved value from the connection field has to be an iterable or "
            "instance of {}. Received \"{}\"".format(connection_type, resolved)
        )

        result = connection_from_array(
            resolved,
            args,
            connection_type=partial(connection_adapter, connection_type),
            edge_type=connection_type.Edge,
            page_info_type=page_info_adapter,
        )
        result.iterable = resolved
        return result

    @classmethod
    def connection_resolver(cls, resolver, connection_type, root, info, **args):
        resolved = resolver(root, info, **args)

        if isinstance(connection_type, NonNull):
            connection_type = connection_type.of_type

        adapter = partial(cls.resolve_connection, connection_type, args)
        return maybe_thenable(resolved, adapter)

    def wrap_resolve(self, parent_resolver):
        resolver = super(IterableConnectionField, self).wrap_resolve(parent_resolver)
        return partial(self.connection_resolver, resolver, self.type)


ConnectionField = IterableConnectionField