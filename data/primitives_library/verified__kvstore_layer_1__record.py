import struct, zlib

MAGIC = b"KVR1"
FLAG_TOMBSTONE = 0x01
_KNOWN_FLAGS = FLAG_TOMBSTONE
_PREFIX = struct.Struct(">4sI")
_META = struct.Struct(">BII")
_MAX_U32 = 0xFFFFFFFF

__all__ = ["encode", "decode"]


def _as_bytes(name, obj):
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (bytearray, memoryview)):
        return bytes(obj)
    raise TypeError(f"{name} must be bytes-like")


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def encode(key: bytes, value, tombstone: bool = False) -> bytes:
    key = _as_bytes("key", key)

    if tombstone:
        value_bytes = b""
        flags = FLAG_TOMBSTONE
    else:
        if value is None:
            raise TypeError("value must be bytes-like unless tombstone is True")
        value_bytes = _as_bytes("value", value)
        flags = 0

    klen = len(key)
    vlen = len(value_bytes)
    if klen > _MAX_U32:
        raise ValueError("key is too large")
    if vlen > _MAX_U32:
        raise ValueError("value is too large")

    body = _META.pack(flags, klen, vlen) + key + value_bytes
    return _PREFIX.pack(MAGIC, _crc32(body)) + body


def decode(buf: bytes, offset: int = 0) -> tuple:
    if offset < 0:
        raise ValueError("offset must be non-negative")

    data_len = len(buf)
    if data_len - offset < _PREFIX.size:
        raise ValueError("truncated record header")

    magic, expected_crc = _PREFIX.unpack_from(buf, offset)
    if magic != MAGIC:
        raise ValueError("bad magic")

    body_start = offset + _PREFIX.size
    if data_len - body_start < _META.size:
        raise ValueError("truncated record metadata")

    flags, klen, vlen = _META.unpack_from(buf, body_start)
    if flags & ~_KNOWN_FLAGS:
        raise ValueError("unknown record flags")

    payload_start = body_start + _META.size
    key_start = payload_start
    key_end = key_start + klen
    value_start = key_end
    value_end = value_start + vlen

    if key_end < key_start or value_end < value_start or value_end > data_len:
        raise ValueError("truncated record payload")

    body = buf[body_start:value_end]
    actual_crc = _crc32(body)
    if actual_crc != expected_crc:
        raise ValueError("CRC mismatch")

    key = bytes(buf[key_start:key_end])
    tombstone = bool(flags & FLAG_TOMBSTONE)
    if tombstone:
        value = None
    else:
        value = bytes(buf[value_start:value_end])

    return key, value, tombstone, value_end