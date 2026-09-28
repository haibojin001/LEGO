import datetime
import struct
from collections import namedtuple

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


class ExtType(namedtuple("ExtType", ("code", "data"))):
    """A MessagePack extension value."""

    def __new__(cls, code, data):
        if not isinstance(code, int):
            raise TypeError("code must be int")
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if code < 0 or code > 127:
            raise ValueError("code must be 0~127")
        return super().__new__(cls, code, data)


class Timestamp:
    """The MessagePack timestamp extension payload representation."""

    __slots__ = ["seconds", "nanoseconds"]

    def __init__(self, seconds, nanoseconds=0):
        if not isinstance(seconds, int):
            raise TypeError("seconds must be an integer")
        if not isinstance(nanoseconds, int):
            raise TypeError("nanoseconds must be an integer")
        if nanoseconds < 0 or nanoseconds >= 1000000000:
            raise ValueError(
                "nanoseconds must be a non-negative integer less than 999999999."
            )
        self.seconds = seconds
        self.nanoseconds = nanoseconds

    def __repr__(self):
        return "Timestamp(seconds={}, nanoseconds={})".format(
            self.seconds, self.nanoseconds
        )

    def __eq__(self, other):
        if type(other) is self.__class__:
            return (
                self.seconds == other.seconds
                and self.nanoseconds == other.nanoseconds
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.seconds, self.nanoseconds))

    @staticmethod
    def from_bytes(b):
        size = len(b)

        if size == 4:
            seconds = struct.unpack("!L", b)[0]
            nanoseconds = 0
        elif size == 8:
            packed = struct.unpack("!Q", b)[0]
            seconds = packed & 0x00000003FFFFFFFF
            nanoseconds = packed >> 34
        elif size == 12:
            nanoseconds, seconds = struct.unpack("!Iq", b)
        else:
            raise ValueError(
                "Timestamp type can only be created from 32, 64, or 96-bit byte objects"
            )

        return Timestamp(seconds, nanoseconds)

    def to_bytes(self):
        seconds = self.seconds
        nanoseconds = self.nanoseconds

        if seconds >> 34 == 0:
            combined = (nanoseconds << 34) | seconds
            if combined & 0xFFFFFFFF00000000 == 0:
                return struct.pack("!L", combined)
            return struct.pack("!Q", combined)

        return struct.pack("!Iq", nanoseconds, seconds)

    @staticmethod
    def from_unix(unix_sec):
        seconds = int(unix_sec // 1)
        nanoseconds = int((unix_sec % 1) * 1000000000)
        return Timestamp(seconds, nanoseconds)

    def to_unix(self):
        return self.seconds + self.nanoseconds / 1e9

    @staticmethod
    def from_unix_nano(unix_ns):
        return Timestamp(*divmod(unix_ns, 1000000000))

    def to_unix_nano(self):
        return self.seconds * 1000000000 + self.nanoseconds

    def to_datetime(self):
        return _EPOCH + datetime.timedelta(
            seconds=self.seconds,
            microseconds=self.nanoseconds // 1000,
        )

    @staticmethod
    def from_datetime(dt):
        if dt.tzinfo is None:
            dt = dt.astimezone()

        offset = dt - _EPOCH
        return Timestamp(
            offset.days * 86400 + offset.seconds,
            offset.microseconds * 1000,
        )