from typing import TYPE_CHECKING

from ..utils.deprecated import warn_deprecation
from ..utils.get_unbound_function import get_unbound_function
from ..utils.props import props
from .field import Field
from .interface import Interface
from .objecttype import ObjectType, ObjectTypeOptions
from .utils import yank_fields_from_attrs

if TYPE_CHECKING:
    from typing import Callable, Dict, Iterable, Type

    from .argument import Argument


class MutationOptions(ObjectTypeOptions):
    arguments = None
    output = None
    resolver = None
    interfaces = ()


class Mutation(ObjectType):
    @classmethod
    def __init_subclass_with_meta__(
        cls,
        interfaces=(),
        resolver=None,
        output=None,
        arguments=None,
        _meta=None,
        **options,
    ):
        if _meta is None:
            _meta = MutationOptions(cls)

        output = output or getattr(cls, "Output", None)
        fields = {}

        for interface in interfaces:
            assert issubclass(
                interface, Interface
            ), (
                f'All interfaces of {cls.__name__} must be a subclass of Interface. '
                f'Received "{interface}".'
            )
            fields.update(interface._meta.fields)

        if not output:
            fields = {}
            for base in reversed(cls.__mro__):
                fields.update(yank_fields_from_attrs(base.__dict__, _as=Field))
            output = cls

        if not arguments:
            input_class = getattr(cls, "Arguments", None)
            if not input_class:
                input_class = getattr(cls, "Input", None)
                if input_class:
                    warn_deprecation(
                        f"Please use {cls.__name__}.Arguments instead of {cls.__name__}.Input."
                        " Input is now only used in ClientMutationID.\n"
                        "Read more:"
                        " https://github.com/graphql-python/graphene/blob/v2.0.0/"
                        "UPGRADE-v2.0.md#mutation-input"
                    )
            arguments = props(input_class) if input_class else {}

        if not resolver:
            mutate = getattr(cls, "mutate", None)
            assert mutate, "All mutations must define a mutate method in it"
            resolver = get_unbound_function(mutate)

        if _meta.fields:
            _meta.fields.update(fields)
        else:
            _meta.fields = fields

        _meta.interfaces = interfaces
        _meta.output = output
        _meta.resolver = resolver
        _meta.arguments = arguments

        super(Mutation, cls).__init_subclass_with_meta__(_meta=_meta, **options)

    @classmethod
    def Field(
        cls, name=None, description=None, deprecation_reason=None, required=False
    ):
        return Field(
            cls._meta.output,
            args=cls._meta.arguments,
            resolver=cls._meta.resolver,
            name=name,
            description=description or cls._meta.description,
            deprecation_reason=deprecation_reason,
            required=required,
        )