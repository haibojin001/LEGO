from typing import TYPE_CHECKING

from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType

if TYPE_CHECKING:
    from typing import Iterable, Type

    from .objecttype import ObjectType


class UnionOptions(BaseOptions):
    types = ()  # type: Iterable[Type[ObjectType]]


class Union(UnmountedType, BaseType):
    @classmethod
    def __init_subclass_with_meta__(cls, types=None, _meta=None, **options):
        assert isinstance(types, (list, tuple)) and len(types) > 0, (
            f"Must provide types for Union {cls.__name__}."
        )

        if not _meta:
            _meta = UnionOptions(cls)

        _meta.types = types
        super(Union, cls).__init_subclass_with_meta__(_meta=_meta, **options)

    @classmethod
    def get_type(cls):
        return cls

    @classmethod
    def resolve_type(cls, instance, info):
        from .objecttype import ObjectType

        if isinstance(instance, ObjectType):
            return type(instance)