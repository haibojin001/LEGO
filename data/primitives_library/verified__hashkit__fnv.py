"""FNV-1a non-cryptographic hash functions."""

__all__ = ["fnv1a_32", "fnv1a_64"]

_FNV1A_32_OFFSET_BASIS = 0x811C9DC5
_FNV1A_32_PRIME = 0x01000193
_FNV1A_32_MASK = 0xFFFFFFFF

_FNV1A_64_OFFSET_BASIS = 0xCBF29CE484222325
_FNV1A_64_PRIME = 0x00000100000001B3
_FNV1A_64_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_32(data: bytes) -> int:
    """Return the 32-bit FNV-1a hash of *data* as an unsigned integer."""
    h = _FNV1A_32_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * _FNV1A_32_PRIME) & _FNV1A_32_MASK
    return h


def fnv1a_64(data: bytes) -> int:
    """Return the 64-bit FNV-1a hash of *data* as an unsigned integer."""
    h = _FNV1A_64_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * _FNV1A_64_PRIME) & _FNV1A_64_MASK
    return h