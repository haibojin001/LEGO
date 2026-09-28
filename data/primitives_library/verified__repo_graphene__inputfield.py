from graphql import Undefined

from .mountedtype import MountedType
from .structures import NonNull
from .utils import get_type


class InputField(MountedType):
    def __init__(
        self,
        type_,
        name=None,
        default_value=Undefined,
        deprecation_reason=None,
        description=None,
        required=False,
        _creation_counter=None,
        **extra_args,
    ):
        MountedType.__init__(self, _creation_counter=_creation_counter)
        self.name = name

        if required:
            assert deprecation_reason is None, (
                f"InputField {name} is required, cannot deprecate it."
            )
            type_ = NonNull(type_)

        self._type = type_
        self.deprecation_reason = deprecation_reason
        self.default_value = default_value
        self.description = description

    @property
    def type(self):
        return get_type(self._type)