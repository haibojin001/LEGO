import struct
import sys
from datetime import datetime as _DateTime

if hasattr(sys, "pypy_version_info"):
    from __pypy__ import newlist_hint
    from __pypy__.builders import BytesBuilder

    _USING_STRINGBUILDER = True

    class BytesIO:
        def __init__(self, s=b""):
            self.builder = BytesBuilder(len(s)) if s else BytesBuilder()
            if s:
                self.builder.append(s)

        def write(self, s):
            if isinstance(s, memoryview):
                s = s.tobytes()
            elif isinstance(s, bytearray):
                s = bytes(s)
            self.builder.append(s)

        def getvalue(self):
            return self.builder.build()

else:
    from io import BytesIO

    _USING_STRINGBUILDER = False

    def newlist_hint(size):
        return []


if sys.version_info >= (3, 15):
    import builtins

    frozendict = builtins.frozendict
else:

    class frozendict:
        pass


from .exceptions import BufferFull, ExtraData, FormatError, OutOfData, StackError
from .ext import ExtType, Timestamp

EX_SKIP = 0
EX_CONSTRUCT = 1
EX_READ_ARRAY_HEADER = 2
EX_READ_MAP_HEADER = 3

TYPE_IMMEDIATE = 0
TYPE_ARRAY = 1
TYPE_MAP = 2
TYPE_RAW = 3
TYPE_BIN = 4
TYPE_EXT = 5

DEFAULT_RECURSE_LIMIT = 1024

_NO_FORMAT_USED = ""
_MSGPACK_HEADERS = {
    0xC4: (1, _NO_FORMAT_USED, TYPE_BIN),
    0xC5: (2, ">H", TYPE_BIN),
    0xC6: (4, ">I", TYPE_BIN),
    0xC7: (2, "Bb", TYPE_EXT),
    0xC8: (3, ">Hb", TYPE_EXT),
    0xC9: (5, ">Ib", TYPE_EXT),
    0xCA: (4, ">f"),
    0xCB: (8, ">d"),
    0xCC: (1, _NO_FORMAT_USED),
    0xCD: (2, ">H"),
    0xCE: (4, ">I"),
    0xCF: (8, ">Q"),
    0xD0: (1, "b"),
    0xD1: (2, ">h"),
    0xD2: (4, ">i"),
    0xD3: (8, ">q"),
    0xD4: (1, "b1s", TYPE_EXT),
    0xD5: (2, "b2s", TYPE_EXT),
    0xD6: (4, "b4s", TYPE_EXT),
    0xD7: (8, "b8s", TYPE_EXT),
    0xD8: (16, "b16s", TYPE_EXT),
    0xD9: (1, _NO_FORMAT_USED, TYPE_RAW),
    0xDA: (2, ">H", TYPE_RAW),
    0xDB: (4, ">I", TYPE_RAW),
    0xDC: (2, ">H", TYPE_ARRAY),
    0xDD: (4, ">I", TYPE_ARRAY),
    0xDE: (2, ">H", TYPE_MAP),
    0xDF: (4, ">I", TYPE_MAP),
}


def _check_type_strict(obj, t, type=type, tuple=tuple):
    if type(t) is tuple:
        return type(obj) in t
    return type(obj) is t


def _get_data_from_buffer(obj):
    view = memoryview(obj)
    if view.itemsize != 1:
        raise ValueError("cannot unpack from multi-byte object")
    return view


def unpackb(packed, **kwargs):
    unpacker = Unpacker(None, max_buffer_size=len(packed), **kwargs)
    unpacker.feed(packed)
    try:
        value = unpacker._unpack()
    except OutOfData:
        raise ValueError("Unpack failed: incomplete input")
    except RecursionError:
        raise StackError
    if unpacker._got_extradata():
        raise ExtraData(value, unpacker._get_extradata())
    return value


class Unpacker:
    def __init__(
        self,
        file_like=None,
        *,
        read_size=0,
        use_list=True,
        raw=False,
        timestamp=0,
        strict_map_key=True,
        object_hook=None,
        object_pairs_hook=None,
        list_hook=None,
        unicode_errors=None,
        max_buffer_size=100 * 1024 * 1024,
        ext_hook=ExtType,
        max_str_len=-1,
        max_bin_len=-1,
        max_array_len=-1,
        max_map_len=-1,
        max_ext_len=-1,
    ):
        if unicode_errors is None:
            unicode_errors = "strict"
        if file_like is None:
            self._feeding = True
        else:
            if not callable(file_like.read):
                raise TypeError("`file_like.read` must be callable")
            self.file_like = file_like
            self._feeding = False

        self._buffer = bytearray()
        self._buff_i = 0
        self._buf_checkpoint = 0

        if not max_buffer_size:
            max_buffer_size = 2**31 - 1
        if max_str_len == -1:
            max_str_len = max_buffer_size
        if max_bin_len == -1:
            max_bin_len = max_buffer_size
        if max_array_len == -1:
            max_array_len = max_buffer_size
        if max_map_len == -1:
            max_map_len = max_buffer_size // 2
        if max_ext_len == -1:
            max_ext_len = max_buffer_size

        self._max_buffer_size = max_buffer_size
        if read_size > max_buffer_size:
            raise ValueError("read_size must be smaller than max_buffer_size")
        self._read_size = read_size or min(max_buffer_size, 16 * 1024)
        self._raw = bool(raw)
        self._strict_map_key = bool(strict_map_key)
        self._unicode_errors = unicode_errors
        self._use_list = use_list
        self._timestamp = timestamp
        self._object_hook = object_hook
        self._object_pairs_hook = object_pairs_hook
        self._list_hook = list_hook
        self._ext_hook = ext_hook
        self._max_str_len = max_str_len
        self._max_bin_len = max_bin_len
        self._max_array_len = max_array_len
        self._max_map_len = max_map_len
        self._max_ext_len = max_ext_len

    def feed(self, next_bytes):
        if not self._feeding:
            raise AssertionError("cannot use feed() with file_like")
        view = _get_data_from_buffer(next_bytes)
        data = view.tobytes()
        if len(self._buffer) - self._buff_i + len(data) > self._max_buffer_size:
            raise BufferFull
        self._buffer.extend(data)

    def _compact(self):
        if self._buff_i:
            del self._buffer[: self._buff_i]
            self._buff_i = 0
            self._buf_checkpoint = 0

    def _read_from_file(self):
        if self._feeding:
            raise OutOfData
        self._compact()
        chunk = self.file_like.read(self._read_size)
        if chunk is None:
            chunk = b""
        view = _get_data_from_buffer(chunk)
        chunk = view.tobytes()
        if not chunk:
            raise OutOfData
        if len(self._buffer) + len(chunk) > self._max_buffer_size:
            raise BufferFull
        self._buffer.extend(chunk)

    def _reserve(self, n):
        while len(self._buffer) - self._buff_i < n:
            if self._feeding:
                raise OutOfData
            self._read_from_file()

    def _take(self, n):
        self._reserve(n)
        start = self._buff_i
        self._buff_i += n
        return bytes(self._buffer[start : start + n])

    def _byte(self):
        self._reserve(1)
        value = self._buffer[self._buff_i]
        self._buff_i += 1
        return value

    def _length(self, nbytes):
        if nbytes == 1:
            return self._byte()
        return struct.unpack({2: ">H", 4: ">I"}[nbytes], self._take(nbytes))[0]

    def _check_size(self, size, maximum, kind):
        if size > maximum:
            raise ValueError("%s exceeds max_%s_len(%s)" % (kind, kind, maximum))

    def _decode_raw(self, size, construct):
        self._check_size(size, self._max_str_len, "str")
        data = self._take(size)
        if not construct:
            return None
        if self._raw:
            return data
        return data.decode("utf-8", self._unicode_errors)

    def _decode_bin(self, size, construct):
        self._check_size(size, self._max_bin_len, "bin")
        data = self._take(size)
        return data if construct else None

    def _decode_ext(self, size, code, construct):
        self._check_size(size, self._max_ext_len, "ext")
        data = self._take(size)
        if not construct:
            return None
        if code == -1:
            stamp = Timestamp.from_bytes(data)
            if self._timestamp == 0:
                return stamp
            if self._timestamp == 1:
                return stamp.to_unix()
            if self._timestamp == 2:
                return stamp.to_unix_nano()
            if self._timestamp == 3:
                return stamp.to_datetime()
        return self._ext_hook(code, data)

    def _array(self, size, depth, construct):
        self._check_size(size, self._max_array_len, "array")
        if depth >= DEFAULT_RECURSE_LIMIT:
            raise StackError
        if not construct:
            for _ in range(size):
                self._parse(depth + 1, False)
            return None
        values = newlist_hint(size)
        for _ in range(size):
            values.append(self._parse(depth + 1, True))
        if self._list_hook is not None:
            return self._list_hook(values)
        if self._use_list:
            return values
        return tuple(values)

    def _map(self, size, depth, construct):
        self._check_size(size, self._max_map_len, "map")
        if depth >= DEFAULT_RECURSE_LIMIT:
            raise StackError
        if not construct:
            for _ in range(size * 2):
                self._parse(depth + 1, False)
            return None

        pairs = newlist_hint(size)
        for _ in range(size):
            key = self._parse(depth + 1, True)
            if self._strict_map_key and not _check_type_strict(key, (str, bytes)):
                raise ValueError("%s is not allowed for map key when strict_map_key=True" % type(key))
            value = self._parse(depth + 1, True)
            pairs.append((key, value))

        if self._object_pairs_hook is not None:
            return self._object_pairs_hook(pairs)
        result = {}
        for key, value in pairs:
            result[key] = value
        if self._object_hook is not None:
            return self._object_hook(result)
        return result

    def _parse(self, depth=0, construct=True):
        code = self._byte()

        if code <= 0x7F:
            return code if construct else None
        if 0x80 <= code <= 0x8F:
            return self._map(code & 0x0F, depth, construct)
        if 0x90 <= code <= 0x9F:
            return self._array(code & 0x0F, depth, construct)
        if 0xA0 <= code <= 0xBF:
            return self._decode_raw(code & 0x1F, construct)
        if code >= 0xE0:
            return code - 256 if construct else None

        if code == 0xC0:
            return None
        if code == 0xC1:
            raise FormatError("Unknown code: c1")
        if code == 0xC2:
            return False if construct else None
        if code == 0xC3:
            return True if construct else None

        if code == 0xC4:
            return self._decode_bin(self._length(1), construct)
        if code == 0xC5:
            return self._decode_bin(self._length(2), construct)
        if code == 0xC6:
            return self._decode_bin(self._length(4), construct)
        if code in (0xC7, 0xC8, 0xC9):
            length = self._length({0xC7: 1, 0xC8: 2, 0xC9: 4}[code])
            ext_code = struct.unpack("b", self._take(1))[0]
            return self._decode_ext(length, ext_code, construct)
        if code == 0xCA:
            value = struct.unpack(">f", self._take(4))[0]
            return value if construct else None
        if code == 0xCB:
            value = struct.unpack(">d", self._take(8))[0]
            return value if construct else None
        if code == 0xCC:
            value = self._byte()
            return value if construct else None
        if code == 0xCD:
            value = struct.unpack(">H", self._take(2))[0]
            return value if construct else None
        if code == 0xCE:
            value = struct.unpack(">I", self._take(4))[0]
            return value if construct else None
        if code == 0xCF:
            value = struct.unpack(">Q", self._take(8))[0]
            return value if construct else None
        if code == 0xD0:
            value = struct.unpack("b", self._take(1))[0]
            return value if construct else None
        if code == 0xD1:
            value = struct.unpack(">h", self._take(2))[0]
            return value if construct else None
        if code == 0xD2:
            value = struct.unpack(">i", self._take(4))[0]
            return value if construct else None
        if code == 0xD3:
            value = struct.unpack(">q", self._take(8))[0]
            return value if construct else None
        if 0xD4 <= code <= 0xD8:
            size = (1, 2, 4, 8, 16)[code - 0xD4]
            ext_code = struct.unpack("b", self._take(1))[0]
            return self._decode_ext(size, ext_code, construct)
        if code == 0xD9:
            return self._decode_raw(self._length(1), construct)
        if code == 0xDA:
            return self._decode_raw(self._length(2), construct)
        if code == 0xDB:
            return self._decode_raw(self._length(4), construct)
        if code == 0xDC:
            return self._array(self._length(2), depth, construct)
        if code == 0xDD:
            return self._array(self._length(4), depth, construct)
        if code == 0xDE:
            return self._map(self._length(2), depth, construct)
        if code == 0xDF:
            return self._map(self._length(4), depth, construct)
        raise FormatError("Unknown code: %02x" % code)

    def _unpack(self, execute=EX_CONSTRUCT):
        checkpoint = self._buff_i
        self._buf_checkpoint = checkpoint
        try:
            if execute == EX_SKIP:
                return self._parse(0, False)
            if execute == EX_READ_ARRAY_HEADER:
                code = self._byte()
                if 0x90 <= code <= 0x9F:
                    return code & 0x0F
                if code == 0xDC:
                    return self._length(2)
                if code == 0xDD:
                    return self._length(4)
                raise FormatError("expected array")
            if execute == EX_READ_MAP_HEADER:
                code = self._byte()
                if 0x80 <= code <= 0x8F:
                    return code & 0x0F
                if code == 0xDE:
                    return self._length(2)
                if code == 0xDF:
                    return self._length(4)
                raise FormatError("expected map")
            return self._parse()
        except OutOfData:
            self._buff_i = checkpoint
            raise

    def __iter__(self):
        return self

    def __next__(self):
        self._compact()
        try:
            return self._unpack()
        except OutOfData:
            raise StopIteration

    def skip(self):
        self._compact()
        self._unpack(EX_SKIP)

    def read_array_header(self):
        self._compact()
        return self._unpack(EX_READ_ARRAY_HEADER)

    def read_map_header(self):
        self._compact()
        return self._unpack(EX_READ_MAP_HEADER)

    def read_bytes(self, n):
        self._compact()
        checkpoint = self._buff_i
        try:
            return self._take(n)
        except OutOfData:
            self._buff_i = checkpoint
            raise

    def _got_extradata(self):
        return self._buff_i < len(self._buffer)

    def _get_extradata(self):
        return bytes(self._buffer[self._buff_i :])


class Packer:
    def __init__(
        self,
        default=None,
        use_single_float=False,
        autoreset=True,
        use_bin_type=True,
        strict_types=False,
        datetime=False,
        unicode_errors="strict",
        buf_size=0,
    ):
        self._default = default
        self._use_single_float = use_single_float
        self._autoreset = autoreset
        self._use_bin_type = use_bin_type
        self._strict_types = strict_types
        self._datetime = datetime
        self._unicode_errors = unicode_errors
        self._buffer = BytesIO()

    def _write(self, data):
        self._buffer.write(data)

    def _pack_integer(self, value):
        if value >= 0:
            if value <= 0x7F:
                self._write(bytes((value,)))
            elif value <= 0xFF:
                self._write(b"\xcc" + struct.pack("B", value))
            elif value <= 0xFFFF:
                self._write(b"\xcd" + struct.pack(">H", value))
            elif value <= 0xFFFFFFFF:
                self._write(b"\xce" + struct.pack(">I", value))
            elif value <= 0xFFFFFFFFFFFFFFFF:
                self._write(b"\xcf" + struct.pack(">Q", value))
            else:
                raise OverflowError("Integer value out of range")
        else:
            if value >= -32:
                self._write(bytes((value & 0xFF,)))
            elif value >= -128:
                self._write(b"\xd0" + struct.pack("b", value))
            elif value >= -32768:
                self._write(b"\xd1" + struct.pack(">h", value))
            elif value >= -2147483648:
                self._write(b"\xd2" + struct.pack(">i", value))
            elif value >= -9223372036854775808:
                self._write(b"\xd3" + struct.pack(">q", value))
            else:
                raise OverflowError("Integer value out of range")

    def _pack_raw_header(self, size):
        if size <= 31:
            self._write(bytes((0xA0 | size,)))
        elif size <= 0xFF:
            self._write(b"\xd9" + struct.pack("B", size))
        elif size <= 0xFFFF:
            self._write(b"\xda" + struct.pack(">H", size))
        elif size <= 0xFFFFFFFF:
            self._write(b"\xdb" + struct.pack(">I", size))
        else:
            raise OverflowError("String is too large")

    def _pack_bin_header(self, size):
        if size <= 0xFF:
            self._write(b"\xc4" + struct.pack("B", size))
        elif size <= 0xFFFF:
            self._write(b"\xc5" + struct.pack(">H", size))
        elif size <= 0xFFFFFFFF:
            self._write(b"\xc6" + struct.pack(">I", size))
        else:
            raise OverflowError("Bytes object is too large")

    def _pack_array_header(self, size):
        if size <= 15:
            self._write(bytes((0x90 | size,)))
        elif size <= 0xFFFF:
            self._write(b"\xdc" + struct.pack(">H", size))
        elif size <= 0xFFFFFFFF:
            self._write(b"\xdd" + struct.pack(">I", size))
        else:
            raise OverflowError("Array is too large")

    def _pack_map_header(self, size):
        if size <= 15:
            self._write(bytes((0x80 | size,)))
        elif size <= 0xFFFF:
            self._write(b"\xde" + struct.pack(">H", size))
        elif size <= 0xFFFFFFFF:
            self._write(b"\xdf" + struct.pack(">I", size))
        else:
            raise OverflowError("Dict is too large")

    def _pack_ext(self, code, data):
        data = bytes(data)
        size = len(data)
        if not -128 <= code <= 127:
            raise ValueError("code must be an integer between -128 and 127")
        if size == 1:
            self._write(b"\xd4" + struct.pack("b", code))
        elif size == 2:
            self._write(b"\xd5" + struct.pack("b", code))
        elif size == 4:
            self._write(b"\xd6" + struct.pack("b", code))
        elif size == 8:
            self._write(b"\xd7" + struct.pack("b", code))
        elif size == 16:
            self._write(b"\xd8" + struct.pack("b", code))
        elif size <= 0xFF:
            self._write(b"\xc7" + struct.pack("Bb", size, code))
        elif size <= 0xFFFF:
            self._write(b"\xc8" + struct.pack(">Hb", size, code))
        elif size <= 0xFFFFFFFF:
            self._write(b"\xc9" + struct.pack(">Ib", size, code))
        else:
            raise OverflowError("ExtType data is too large")
        self._write(data)

    def _pack(self, obj, depth=0):
        if depth >= DEFAULT_RECURSE_LIMIT:
            raise StackError

        typ = type(obj)
        if obj is None:
            self._write(b"\xc0")
        elif typ is bool:
            self._write(b"\xc3" if obj else b"\xc2")
        elif isinstance(obj, int) and (not self._strict_types or typ is int):
            self._pack_integer(obj)
        elif isinstance(obj, float) and (not self._strict_types or typ is float):
            self._write(struct.pack(">f" if self._use_single_float else ">d", obj))
            if self._use_single_float:
                current = self._buffer.getvalue()
                self._buffer = BytesIO(current[:-4])
                self._write(b"\xca" + current[-4:])
            else:
                current = self._buffer.getvalue()
                self._buffer = BytesIO(current[:-8])
                self._write(b"\xcb" + current[-8:])
        elif isinstance(obj, str) and (not self._strict_types or typ is str):
            data = obj.encode("utf-8", self._unicode_errors)
            self._pack_raw_header(len(data))
            self._write(data)
        elif isinstance(obj, (bytes, bytearray, memoryview)) and (
            not self._strict_types or typ is bytes
        ):
            data = bytes(obj)
            if self._use_bin_type:
                self._pack_bin_header(len(data))
            else:
                self._pack_raw_header(len(data))
            self._write(data)
        elif isinstance(obj, Timestamp) and (not self._strict_types or typ is Timestamp):
            self._pack_ext(-1, obj.to_bytes())
        elif self._datetime and isinstance(obj, _DateTime) and (
            not self._strict_types or typ is _DateTime
        ):
            self._pack_ext(-1, Timestamp.from_datetime(obj).to_bytes())
        elif isinstance(obj, ExtType) and (not self._strict_types or typ is ExtType):
            self._pack_ext(obj.code, obj.data)
        elif isinstance(obj, (list, tuple)) and (
            not self._strict_types or typ is list
        ):
            self._pack_array_header(len(obj))
            for item in obj:
                self._pack(item, depth + 1)
        elif isinstance(obj, dict) and (not self._strict_types or typ is dict):
            self._pack_map_header(len(obj))
            for key, value in obj.items():
                self._pack(key, depth + 1)
                self._pack(value, depth + 1)
        else:
            if self._default is None:
                raise TypeError("can not serialize %r object" % typ.__name__)
            self._pack(self._default(obj), depth + 1)

    def pack(self, obj):
        self._pack(obj)
        if self._autoreset:
            return self.bytes()
        return None

    def pack_map_pairs(self, pairs):
        pairs = list(pairs)
        self._pack_map_header(len(pairs))
        for key, value in pairs:
            self._pack(key)
            self._pack(value)
        if self._autoreset:
            return self.bytes()
        return None

    def pack_array_header(self, size):
        self._pack_array_header(size)
        if self._autoreset:
            return self.bytes()
        return None

    def pack_map_header(self, size):
        self._pack_map_header(size)
        if self._autoreset:
            return self.bytes()
        return None

    def pack_ext_type(self, typecode, data):
        self._pack_ext(typecode, data)
        if self._autoreset:
            return self.bytes()
        return None

    def bytes(self):
        value = self._buffer.getvalue()
        self._buffer = BytesIO()
        return value

    def reset(self):
        self._buffer = BytesIO()

    def getbuffer(self):
        return memoryview(self._buffer.getvalue())


def packb(o, **kwargs):
    return Packer(**kwargs).pack(o)