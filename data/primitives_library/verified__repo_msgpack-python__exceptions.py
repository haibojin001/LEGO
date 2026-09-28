class UnpackException(Exception):
    """Base class for some exceptions raised while unpacking.

    NOTE: unpack may raise exception other than subclass of
    UnpackException. If you want to catch all errors, catch
    Exception instead.
    """


class BufferFull(UnpackException):
    pass


class OutOfData(UnpackException):
    pass


class FormatError(ValueError, UnpackException):
    """Invalid msgpack format."""


class StackError(ValueError, UnpackException):
    """Too nested."""


UnpackValueError = ValueError


class ExtraData(UnpackValueError):
    """Raised when one-shot unpacking encounters trailing data."""

    def __init__(self, unpacked, extra):
        self.unpacked = unpacked
        self.extra = extra

    def __str__(self):
        return "unpack(b) received extra data."


PackException = Exception
PackValueError = ValueError
PackOverflowError = OverflowError