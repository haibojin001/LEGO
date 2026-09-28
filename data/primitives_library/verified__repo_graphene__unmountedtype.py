from ..utils.orderedtype import OrderedType


class UnmountedType(OrderedType):
    """
    Proxy base class for Graphene types that can be mounted dynamically.
    """

    def __init__(self, *args, **kwargs):
        super(UnmountedType, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def get_type(self):
        """
        Return the type represented by this unmounted instance.
        """
        raise NotImplementedError(f"get_type not implemented in {self}")

    def mount_as(self, _as):
        return _as.mounted(self)

    def Field(self):  # noqa: N802
        from .field import Field

        return self.mount_as(Field)

    def InputField(self):  # noqa: N802
        from .inputfield import InputField

        return self.mount_as(InputField)

    def Argument(self):  # noqa: N802
        from .argument import Argument

        return self.mount_as(Argument)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, UnmountedType)
            and self.get_type() == other.get_type()
            and self.args == other.args
            and self.kwargs == other.kwargs
        )