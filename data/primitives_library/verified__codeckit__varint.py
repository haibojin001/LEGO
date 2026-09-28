"""Unsigned LEB128 variable-length integer encoding."""

__all__ = ("encode_uvarint", "decode_uvarint")


def encode_uvarint(n: int) -> bytes:
    """Encode a non-negative integer as unsigned LEB128 bytes."""
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")

    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_uvarint(data: bytes, offset: int = 0) -> tuple:
    """Decode an unsigned LEB128 integer from data starting at offset.

    Returns:
        (value, new_offset)

    Raises:
        ValueError: if the input ends before the varint terminates.
        IndexError: if offset is outside the valid range [0, len(data)].
    """
    if not isinstance(offset, int):
        raise TypeError("offset must be an int")
    if offset < 0 or offset > len(data):
        raise IndexError("offset out of range")

    value = 0
    shift = 0
    pos = offset

    while pos < len(data):
        byte = data[pos]
        pos += 1

        value |= (byte & 0x7F) << shift

        if byte < 0x80:
            return value, pos

        shift += 7

    raise ValueError("incomplete unsigned LEB128 varint")