from .unmountedtype import UnmountedType
from .utils import get_type


class Structure(UnmountedType):
    """Base class for GraphQL wrapper type modifiers."""

    def __init__(self, of_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(of_type, UnmountedType) and not isinstance(of_type, Structure):
            wrapper_name = self.__class__.__name__
            inner_name = of_type.__class__.__name__
            raise Exception(
                f"{wrapper_name} could not have a mounted {inner_name}()"
                f" as inner type. Try with {wrapper_name}({inner_name})."
            )

        self._of_type = of_type

    @property
    def of_type(self):
        return get_type(self._of_type)

    def get_type(self):
        return self


class List(Structure):
    """GraphQL list type modifier."""

    def __str__(self):
        return f"[{self.of_type}]"

    def __eq__(self, other):
        return (
            isinstance(other, List)
            and self.of_type == other.of_type
            and self.args == other.args
            and self.kwargs == other.kwargs
        )


class NonNull(Structure):
    """GraphQL non-null type modifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not isinstance(
            self._of_type, NonNull
        ), f"Can only create NonNull of a Nullable GraphQLType but got: {self._of_type}."

    def __str__(self):
        return f"{self.of_type}!"

    def __eq__(self, other):
        return (
            isinstance(other, NonNull)
            and self.of_type == other.of_type
            and self.args == other.args
            and self.kwargs == other.kwargs
        )