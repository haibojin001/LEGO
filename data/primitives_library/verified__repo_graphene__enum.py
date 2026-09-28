from enum import Enum as PyEnum

from graphene.utils.subclass_with_meta import SubclassWithMeta_Meta

from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType


def eq_enum(self, other):
    if isinstance(other, self.__class__):
        return self is other
    return self.value is other


def hash_enum(self):
    return hash(self.name)


EnumType = type(PyEnum)


class EnumOptions(BaseOptions):
    enum = None
    deprecation_reason = None


class EnumMeta(SubclassWithMeta_Meta):
    def __new__(mcls, name, bases, namespace, **options):
        members = dict(namespace)
        members["__eq__"] = eq_enum
        members["__hash__"] = hash_enum
        members.pop("Meta", None)

        python_enum = PyEnum(mcls.__name__, members)
        result = SubclassWithMeta_Meta.__new__(
            mcls,
            name,
            bases,
            dict(namespace, __enum__=python_enum),
            **options
        )
        globals()[name] = result.__enum__
        return result

    def get(cls, value):
        return cls._meta.enum(value)

    def __getitem__(cls, value):
        return cls._meta.enum[value]

    def __prepare__(name, bases, **kwargs):
        return {}

    def __call__(cls, *args, **kwargs):
        if cls is Enum:
            description = kwargs.pop("description", None)
            deprecation_reason = kwargs.pop("deprecation_reason", None)
            enum = PyEnum(*args, **kwargs)
            return cls.from_enum(
                enum,
                description=description,
                deprecation_reason=deprecation_reason,
            )
        return super(EnumMeta, cls).__call__(*args, **kwargs)

    def __iter__(cls):
        return iter(cls._meta.enum)

    def from_enum(cls, enum, name=None, description=None, deprecation_reason=None):
        type_name = name or enum.__name__
        type_description = description or enum.__doc__ or "An enumeration."
        meta = type(
            "Meta",
            (object,),
            {
                "enum": enum,
                "description": type_description,
                "deprecation_reason": deprecation_reason,
            },
        )
        return type(type_name, (Enum,), {"Meta": meta})


class Enum(UnmountedType, BaseType, metaclass=EnumMeta):
    """
    GraphQL enumeration type.
    """

    @classmethod
    def __init_subclass_with_meta__(cls, enum=None, _meta=None, **options):
        if _meta is None:
            _meta = EnumOptions(cls)

        _meta.enum = enum or cls.__enum__
        _meta.deprecation_reason = options.pop("deprecation_reason", None)

        for member_name, member in _meta.enum.__members__.items():
            setattr(cls, member_name, member)

        super(Enum, cls).__init_subclass_with_meta__(_meta=_meta, **options)

    @classmethod
    def get_type(cls):
        return cls