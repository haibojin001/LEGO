import inspect
from collections.abc import Mapping
from functools import partial

from .argument import Argument, to_arguments
from .mountedtype import MountedType
from .resolver import default_resolver
from .structures import NonNull
from .unmountedtype import UnmountedType
from .utils import get_type
from ..utils.deprecated import warn_deprecation

base_type = type


def source_resolver(source, root, info, **args):
    value = default_resolver(source, None, root, info, **args)
    if inspect.isfunction(value) or inspect.ismethod(value):
        return value()
    return value


class Field(MountedType):
    def __init__(
        self,
        type_,
        args=None,
        resolver=None,
        source=None,
        deprecation_reason=None,
        name=None,
        description=None,
        required=False,
        _creation_counter=None,
        default_value=None,
        **extra_args,
    ):
        super().__init__(_creation_counter=_creation_counter)

        assert not args or isinstance(
            args, Mapping
        ), f'Arguments in a field have to be a mapping, received "{args}".'
        assert not (
            source and resolver
        ), "A Field cannot have a source and a resolver in at the same time."
        assert not callable(
            default_value
        ), f'The default value can not be a function but received "{base_type(default_value)}".'

        if required:
            type_ = NonNull(type_)

        if isinstance(name, (Argument, UnmountedType)):
            extra_args["name"] = name
            name = None

        if isinstance(source, (Argument, UnmountedType)):
            extra_args["source"] = source
            source = None

        self.name = name
        self._type = type_
        self.args = to_arguments(args or {}, extra_args)
        self.resolver = partial(source_resolver, source) if source else resolver
        self.deprecation_reason = deprecation_reason
        self.description = description
        self.default_value = default_value

    @property
    def type(self):
        return get_type(self._type)

    get_resolver = None

    def wrap_resolve(self, parent_resolver):
        if self.get_resolver is not None:
            warn_deprecation(
                "The get_resolver method is being deprecated, please rename it to wrap_resolve."
            )
            return self.get_resolver(parent_resolver)
        return self.resolver or parent_resolver

    def wrap_subscribe(self, parent_subscribe):
        return parent_subscribe