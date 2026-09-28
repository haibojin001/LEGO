from uuid import UUID as _UUID

from graphql import Undefined
from graphql.error import GraphQLError
from graphql.language.ast import StringValueNode

from .scalars import Scalar


class UUID(Scalar):
    """
    Leverages Python's uuid.UUID implementation to expose UUID values as native
    UUID objects for fields, resolvers, and inputs.
    """

    @staticmethod
    def serialize(uuid):
        if isinstance(uuid, str):
            uuid = _UUID(uuid)

        assert isinstance(uuid, _UUID), f"Expected UUID instance, received {uuid}"
        return str(uuid)

    @staticmethod
    def parse_literal(node, _variables=None):
        if isinstance(node, StringValueNode):
            return _UUID(node.value)
        return Undefined

    @staticmethod
    def parse_value(value):
        if isinstance(value, _UUID):
            return value

        try:
            return _UUID(value)
        except (ValueError, AttributeError):
            raise GraphQLError(f"UUID cannot represent value: {value!r}")