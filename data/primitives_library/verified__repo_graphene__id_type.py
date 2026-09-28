from typing import Type

from graphql_relay import from_global_id, to_global_id

from ..types import ID, UUID
from ..types.base import BaseType


class BaseGlobalIDType:
    """Interface for global identifier encoding strategies."""

    graphene_type: Type[BaseType] = ID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        raise NotImplementedError

    @classmethod
    def to_global_id(cls, _type, _id):
        raise NotImplementedError


class DefaultGlobalIDType(BaseGlobalIDType):
    """Global IDs encoded as base64 ``TypeName:id`` values."""

    graphene_type = ID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        try:
            _type, _id = from_global_id(global_id)
            if not _type:
                raise ValueError("Invalid Global ID")
            return _type, _id
        except Exception as e:
            raise Exception(
                f'Unable to parse global ID "{global_id}". '
                'Make sure it is a base64 encoded string in the format: "TypeName:id". '
                f"Exception message: {e}"
            )

    @classmethod
    def to_global_id(cls, _type, _id):
        return to_global_id(_type, _id)


class SimpleGlobalIDType(BaseGlobalIDType):
    """Global ID strategy that uses the object ID directly."""

    graphene_type = ID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        _type = info.return_type.graphene_type._meta.name
        return _type, global_id

    @classmethod
    def to_global_id(cls, _type, _id):
        return _id


class UUIDGlobalIDType(BaseGlobalIDType):
    """Global ID strategy for UUID identifiers."""

    graphene_type = UUID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        _type = info.return_type.graphene_type._meta.name
        return _type, global_id

    @classmethod
    def to_global_id(cls, _type, _id):
        return _id