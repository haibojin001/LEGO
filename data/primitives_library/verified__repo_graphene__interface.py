from typing import TYPE_CHECKING

from .base import BaseOptions, BaseType
from .field import Field
from .utils import yank_fields_from_attrs

if TYPE_CHECKING:
    from typing import Dict, Iterable, Type


class InterfaceOptions(BaseOptions):
    fields = None
    interfaces = ()


class Interface(BaseType):
    """
    Defines a GraphQL interface.

    Interfaces describe fields shared by multiple object types and may provide a
    resolver for determining the concrete object type returned at runtime.
    """

    @classmethod
    def __init_subclass_with_meta__(cls, _meta=None, interfaces=(), **options):
        metadata = _meta if _meta is not None else InterfaceOptions(cls)

        collected_fields = {}
        for ancestor in reversed(cls.__mro__):
            declared = yank_fields_from_attrs(ancestor.__dict__, _as=Field)
            collected_fields.update(declared)

        if metadata.fields:
            metadata.fields.update(collected_fields)
        else:
            metadata.fields = collected_fields

        if not metadata.interfaces:
            metadata.interfaces = interfaces

        super().__init_subclass_with_meta__(_meta=metadata, **options)

    @classmethod
    def resolve_type(cls, instance, info):
        from .objecttype import ObjectType

        if isinstance(instance, ObjectType):
            return instance.__class__

    def __init__(self, *args, **kwargs):
        raise Exception("An Interface cannot be initialized")