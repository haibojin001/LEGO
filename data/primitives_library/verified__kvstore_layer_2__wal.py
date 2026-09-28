import os

from kvstore.record import (
    FLAG_TOMBSTONE,
    MAGIC,
    _KNOWN_FLAGS,
    _MAX_U32,
    _META,
    _PREFIX,
    decode as _record_decode,
    encode as _record_encode,
)
from kvstore.serialize import from_bytes, to_bytes


_BYTES_TYPES = (bytes, bytearray, memoryview)


def _bytes(value):
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _is_bytes(value):
    return isinstance(value, _BYTES_TYPES)


def _known_flags(flags):
    if not isinstance(flags, int) or isinstance(flags, bool):
        return False
    if flags < 0:
        return False

    try:
        max_u32 = int(_MAX_U32)
    except Exception:
        max_u32 = (1 << 32) - 1
    if flags > max_u32:
        return False

    known = _KNOWN_FLAGS
    if isinstance(known, int):
        return (flags & ~known) == 0

    try:
        return flags in known
    except TypeError:
        return flags == 0 or flags == FLAG_TOMBSTONE


def _flag_to_tombstone(flag):
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, int):
        return bool(flag & FLAG_TOMBSTONE)
    return bool(flag)


def _normalise_decoded(result):
    if isinstance(result, dict):
        key = result.get("key")
        value = result.get("value")
        if "tombstone" in result:
            tombstone = bool(result["tombstone"])
        else:
            tombstone = _flag_to_tombstone(result.get("flags", 0))
        next_offset = result.get("next_offset", result.get("offset", result.get("end")))
        if not _is_bytes(key):
            raise ValueError("decoded record has non-bytes key")
        if value is not None and not _is_bytes(value):
            raise ValueError("decoded record has non-bytes value")
        return _bytes(key), _bytes(value) if value is not None else b"", tombstone, next_offset

    if not isinstance(result, (tuple, list)):
        raise ValueError("decoded record has unsupported shape")

    n = len(result)

    if n >= 4 and _is_bytes(result[0]) and (result[1] is None or _is_bytes(result[1])):
        key, value, flag, next_offset = result[0], result[1], result[2], result[3]
        return _bytes(key), _bytes(value) if value is not None else b"", _flag_to_tombstone(flag), next_offset

    if n >= 4 and isinstance(result[0], int) and _is_bytes(result[1]) and (
        result[2] is None or _is_bytes(result[2])
    ):
        next_offset, key, value, flag = result[0], result[1], result[2], result[3]
        return _bytes(key), _bytes(value) if value is not None else b"", _flag_to_tombstone(flag), next_offset

    if n >= 3 and _is_bytes(result[0]) and (result[1] is None or _is_bytes(result[1])):
        key, value, flag = result[0], result[1], result[2]
        return _bytes(key), _bytes(value) if value is not None else b"", _flag_to_tombstone(flag), None

    raise ValueError("decoded record has unsupported shape")


def _encode_record(key, value_bytes, tombstone):
    flags = FLAG_TOMBSTONE if tombstone else 0

    attempts = (
        lambda: _record_encode(key, value_bytes, tombstone=tombstone),
        lambda: _record_encode(key, value_bytes, flags=flags),
        lambda: _record_encode(key, value_bytes, flags),
        lambda: _record_encode(key, value_bytes, tombstone),
        lambda: _record_encode(key, value_bytes),
    )

    last_error = None
    for attempt in attempts:
        try:
            encoded = attempt()
        except TypeError as exc:
            last_error = exc
            continue
        if not _is_bytes(encoded):
            raise TypeError("record.encode() must return bytes")
        return _bytes(encoded)

    if last_error is not None:
        raise last_error
    raise TypeError("unable to encode record")


def _candidate_totals(data, offset):
    remaining = len(data) - offset
    totals = []

    for meta in (_META, _PREFIX):
        try:
            size = meta.size
        except AttributeError:
            continue

        if remaining < size:
            continue

        try:
            fields = meta.unpack_from(data, offset)
        except Exception:
            continue

        if not isinstance(fields, tuple):
            continue

        magic_positions = [i for i, field in enumerate(fields) if field == MAGIC]

        if magic_positions:
            for magic_index in magic_positions:
                indexes = [i for i in range(len(fields)) if i != magic_index]

                for length_index in indexes:
                    length = fields[length_index]
                    if isinstance(length, int) and not isinstance(length, bool) and length >= 0:
                        if length <= remaining:
                            totals.append(length)
                        if size + length <= remaining:
                            totals.append(size + length)

                for flag_index in indexes:
                    flags = fields[flag_index]
                    if not _known_flags(flags):
                        continue
                    for key_index in indexes:
                        if key_index == flag_index:
                            continue
                        key_len = fields[key_index]
                        if not isinstance(key_len, int) or isinstance(key_len, bool) or key_len < 0:
                            continue
                        for value_index in indexes:
                            if value_index == flag_index or value_index == key_index:
                                continue
                            value_len = fields[value_index]
                            if not isinstance(value_len, int) or isinstance(value_len, bool) or value_len < 0:
                                continue
                            total = size + key_len + value_len
                            if total <= remaining:
                                totals.append(total)
        else:
            for field in fields:
                if isinstance(field, int) and not isinstance(field, bool) and field >= 0:
                    if field <= remaining:
                        totals.append(field)
                    if size + field <= remaining:
                        totals.append(size + field)

    seen = set()
    ordered = []
    for total in totals:
        if total <= 0 or total > remaining or total in seen:
            continue
        seen.add(total)
        ordered.append(total)

    ordered.sort()
    return ordered


def _decode_exact(blob):
    key, value_bytes, tombstone, _ = _normalise_decoded(_record_decode(blob))
    return key, value_bytes, tombstone


def _decode_at(data, offset):
    try:
        key, value_bytes, tombstone, next_offset = _normalise_decoded(_record_decode(data, offset))
        if next_offset is None:
            encoded = _encode_record(key, value_bytes, tombstone)
            next_offset = offset + len(encoded)
        elif isinstance(next_offset, int):
            if offset < next_offset <= len(data):
                pass
            elif 0 < next_offset <= len(data) - offset:
                next_offset = offset + next_offset
            else:
                raise ValueError("invalid next offset")
        else:
            raise ValueError("invalid next offset")

        if next_offset <= offset or next_offset > len(data):
            raise ValueError("invalid next offset")
        return key, value_bytes, tombstone, next_offset
    except TypeError:
        pass
    except Exception:
        pass

    for total in _candidate_totals(data, offset):
        try:
            key, value_bytes, tombstone = _decode_exact(data[offset : offset + total])
        except Exception:
            continue
        return key, value_bytes, tombstone, offset + total

    try:
        key, value_bytes, tombstone, next_offset = _normalise_decoded(_record_decode(data[offset:]))
        if next_offset is None:
            encoded = _encode_record(key, value_bytes, tombstone)
            next_offset = offset + len(encoded)
        elif isinstance(next_offset, int):
            if 0 < next_offset <= len(data) - offset:
                next_offset = offset + next_offset
            elif offset < next_offset <= len(data):
                pass
            else:
                raise ValueError("invalid next offset")
        else:
            raise ValueError("invalid next offset")

        if next_offset <= offset or next_offset > len(data):
            raise ValueError("invalid next offset")
        return key, value_bytes, tombstone, next_offset
    except Exception as exc:
        raise ValueError("unable to decode record") from exc


class WAL:
    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file = open(path, "a+b")

    def append(self, key: bytes, value, tombstone=False) -> int:
        if not _is_bytes(key):
            raise TypeError("key must be bytes")
        key = _bytes(key)

        tombstone = bool(tombstone)
        value_bytes = b"" if tombstone else to_bytes(value)
        if not _is_bytes(value_bytes):
            raise TypeError("serialize.to_bytes() must return bytes")
        value_bytes = _bytes(value_bytes)

        record = _encode_record(key, value_bytes, tombstone)

        self._file.seek(0, os.SEEK_END)
        offset = self._file.tell()
        self._file.write(record)
        self._file.flush()
        os.fsync(self._file.fileno())
        return offset

    def replay(self):
        self._file.flush()

        try:
            with open(self.path, "rb") as file:
                data = file.read()
        except FileNotFoundError:
            return

        offset = 0
        while offset < len(data):
            try:
                key, value_bytes, tombstone, next_offset = _decode_at(data, offset)
                value = None if tombstone else from_bytes(value_bytes)
            except Exception:
                break

            yield key, value, tombstone
            offset = next_offset

    def close(self):
        file = getattr(self, "_file", None)
        if file is not None and not file.closed:
            file.close()