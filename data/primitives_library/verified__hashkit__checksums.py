"""CRC-32 and Adler-32 checksum implementations."""

__all__ = ["crc32", "adler32"]

_CRC32_POLY = 0xEDB88320
_ADLER32_MOD = 65521
_ADLER32_NMAX = 5552


def _make_crc32_table() -> tuple[int, ...]:
    table = []
    for n in range(256):
        c = n
        for _ in range(8):
            if c & 1:
                c = _CRC32_POLY ^ (c >> 1)
            else:
                c >>= 1
        table.append(c & 0xFFFFFFFF)
    return tuple(table)


_CRC32_TABLE = _make_crc32_table()


def crc32(data: bytes) -> int:
    """Return the IEEE 802.3 CRC-32 checksum of *data*.

    The result is identical to ``zlib.crc32(data) & 0xffffffff``.
    """
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def adler32(data: bytes) -> int:
    """Return the Adler-32 checksum of *data*.

    The result is identical to ``zlib.adler32(data) & 0xffffffff``.
    """
    a = 1
    b = 0
    length = len(data)
    index = 0

    while index < length:
        end = min(index + _ADLER32_NMAX, length)
        for byte in data[index:end]:
            a += byte
            b += a
        a %= _ADLER32_MOD
        b %= _ADLER32_MOD
        index = end

    return ((b << 16) | a) & 0xFFFFFFFF