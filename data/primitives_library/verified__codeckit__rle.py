"""Run-length encoding utilities for bytes."""

__all__ = ["rle_encode", "rle_decode"]


def rle_encode(data: bytes) -> list:
    """Encode bytes into a list of ``(byte, count)`` runs.

    Each ``byte`` is an integer in the range 0..255, and each ``count`` is
    an integer greater than or equal to 1.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    if not data:
        return []

    runs = []
    current = data[0]
    count = 1

    for byte in data[1:]:
        if byte == current:
            count += 1
        else:
            runs.append((current, count))
            current = byte
            count = 1

    runs.append((current, count))
    return runs


def rle_decode(runs) -> bytes:
    """Decode a sequence of ``(byte, count)`` runs back into bytes."""
    out = bytearray()

    for run in runs:
        try:
            byte, count = run
        except (TypeError, ValueError):
            raise ValueError("each run must be a pair: (byte, count)")

        if not isinstance(byte, int):
            raise TypeError("run byte must be an int")
        if not 0 <= byte <= 255:
            raise ValueError("run byte must be in range 0..255")
        if not isinstance(count, int):
            raise TypeError("run count must be an int")
        if count < 1:
            raise ValueError("run count must be >= 1")

        out.extend(bytes((byte,)) * count)

    return bytes(out)