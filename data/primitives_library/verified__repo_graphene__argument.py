from itertools import chain

from graphql import Undefined

from .dynamic import Dynamic
from .mountedtype import MountedType
from .structures import NonNull
from .utils import get_type


class Argument(MountedType):
    def __init__(
        self,
        type_,
        default_value=Undefined,
        deprecation_reason=None,
        description=None,
        name=None,
        required=False,
        _creation_counter=None,
    ):
        super().__init__(_creation_counter=_creation_counter)

        if required:
            assert deprecation_reason is None, (
                "Argument {} is required, cannot deprecate it.".format(name)
            )
            type_ = NonNull(type_)

        self.name = name
        self._type = type_
        self.default_value = default_value
        self.description = description
        self.deprecation_reason = deprecation_reason

    @property
    def type(self):
        return get_type(self._type)

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return False

        return (
            self.name == other.name
            and self.type == other.type
            and self.default_value == other.default_value
            and self.description == other.description
            and self.deprecation_reason == other.deprecation_reason
        )


def to_arguments(args, extra_args=None):
    from .field import Field
    from .inputfield import InputField
    from .unmountedtype import UnmountedType

    if extra_args:
        extras = sorted(extra_args.items(), key=lambda item: item[1])
    else:
        extras = []

    result = {}
    for fallback_name, candidate in chain(args.items(), extras):
        if isinstance(candidate, Dynamic):
            candidate = candidate.get_type()
            if candidate is None:
                continue

        if isinstance(candidate, UnmountedType):
            candidate = Argument.mounted(candidate)

        if isinstance(candidate, (InputField, Field)):
            raise ValueError(
                "Expected {} to be Argument, but received {}. "
                "Try using Argument({}).".format(
                    fallback_name, type(candidate).__name__, candidate.type
                )
            )

        if not isinstance(candidate, Argument):
            raise ValueError('Unknown argument "{}".'.format(fallback_name))

        argument_name = fallback_name or candidate.name
        assert argument_name not in result, (
            'More than one Argument have same name "{}".'.format(argument_name)
        )
        result[argument_name] = candidate

    return result