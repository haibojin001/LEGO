from base64 import b64decode, b64encode
from binascii import Error as _Error

from graphql.error import GraphQLError
from graphql.language import StringValueNode, print_ast

from .scalars import Scalar


class Base64(Scalar):
    """
    The `Base64` scalar type represents a base64-encoded String.
    """

    @staticmethod
    def serialize(value):
        if isinstance(value, bytes):
            raw_value = value
        elif isinstance(value, str):
            raw_value = value.encode("utf-8")
        else:
            raw_value = str(value).encode("utf-8")
        return b64encode(raw_value).decode("utf-8")

    @classmethod
    def parse_literal(cls, node, _variables=None):
        if not isinstance(node, StringValueNode):
            raise GraphQLError(
                f"Base64 cannot represent non-string value: {print_ast(node)}"
            )
        return cls.parse_value(node.value)

    @staticmethod
    def parse_value(value):
        if isinstance(value, bytes):
            encoded_value = value
        elif isinstance(value, str):
            encoded_value = value.encode("utf-8")
        else:
            raise GraphQLError(
                f"Base64 cannot represent non-string value: {value!r}"
            )

        try:
            return b64decode(encoded_value, validate=True).decode("utf-8")
        except _Error:
            raise GraphQLError(f"Base64 cannot decode value: {encoded_value!r}")