from typing import TYPE_CHECKING
from dataclasses import field, make_dataclass

from .base import BaseOptions, BaseType, BaseTypeMeta
from .field import Field
from .interface import Interface
from .utils import yank_fields_from_attrs

if TYPE_CHECKING:
    from typing import Dict, Iterable, Type


class ObjectTypeOptions(BaseOptions):
    fields = None
    interfaces = ()


class ObjectTypeMeta(BaseTypeMeta):
    def __new__(cls, name_, bases, namespace, **options):
        class InterObjectType:
            pass

        created_type = super().__new__(
            cls,
            name_,
            (InterObjectType,) + bases,
            namespace,
            **options,
        )

        if created_type._meta:
            attributes = []
            for name, declared_field in created_type._meta.fields.items():
                default = (
                    declared_field.default_value
                    if isinstance(declared_field, Field)
                    else None
                )
                attributes.append(
                    (
                        name,
                        "typing.Any",
                        field(default=default),
                    )
                )

            generated = make_dataclass(name_, attributes, bases=())
            InterObjectType.__init__ = generated.__init__
            InterObjectType.__eq__ = generated.__eq__
            InterObjectType.__repr__ = generated.__repr__

        return created_type


class ObjectType(BaseType, metaclass=ObjectTypeMeta):
    @classmethod
    def __init_subclass_with_meta__(
        cls,
        interfaces=(),
        possible_types=(),
        default_resolver=None,
        _meta=None,
        **options,
    ):
        if not _meta:
            _meta = ObjectTypeOptions(cls)

        collected_fields = {}

        for interface in interfaces:
            assert issubclass(interface, Interface), (
                f'All interfaces of {cls.__name__} must be a subclass of Interface. '
                f'Received "{interface}".'
            )
            collected_fields.update(interface._meta.fields)

        for ancestor in reversed(cls.__mro__):
            collected_fields.update(
                yank_fields_from_attrs(ancestor.__dict__, _as=Field)
            )

        assert not (possible_types and cls.is_type_of), (
            f"{cls.__name__}.Meta.possible_types will cause type collision with "
            f"{cls.__name__}.is_type_of. Please use one or other."
        )

        if _meta.fields:
            _meta.fields.update(collected_fields)
        else:
            _meta.fields = collected_fields

        if not _meta.interfaces:
            _meta.interfaces = interfaces

        _meta.possible_types = possible_types
        _meta.default_resolver = default_resolver

        super(ObjectType, cls).__init_subclass_with_meta__(
            _meta=_meta,
            **options,
        )

    is_type_of = None