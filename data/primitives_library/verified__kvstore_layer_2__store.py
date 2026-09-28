import os

from kvstore.record import _as_bytes, decode as _decode_record, encode as _encode_record
from kvstore.serialize import from_bytes as _value_from_bytes, to_bytes as _value_to_bytes


_DEFAULT_WAL_FILENAME = "wal"

_VALUE_TAG_BYTES = b"\x00KVSTORE:BYTES\x00"
_VALUE_TAG_SERIALIZED = b"\x00KVSTORE:SERIALIZED\x00"


def _ends_with_sep(path: str) -> bool:
    if path.endswith(os.sep):
        return True
    return bool(os.altsep and path.endswith(os.altsep))


def _resolve_wal_path(path: str) -> str:
    path = os.fspath(path)
    if not isinstance(path, str):
        raise TypeError("path must be a str or path-like object returning str")
    if path == "":
        raise ValueError("path must not be empty")

    if os.path.isdir(path):
        return os.path.join(path, _DEFAULT_WAL_FILENAME)

    if os.path.exists(path):
        return path

    if _ends_with_sep(path):
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, _DEFAULT_WAL_FILENAME)

    base = os.path.basename(path)
    _, ext = os.path.splitext(base)
    if ext:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path

    os.makedirs(path, exist_ok=True)
    return os.path.join(path, _DEFAULT_WAL_FILENAME)


def _fsync_file(fileobj) -> None:
    fileobj.flush()
    os.fsync(fileobj.fileno())


def _serialize_value(value):
    if value is None:
        raise TypeError("value must not be None")

    if isinstance(value, (bytes, bytearray, memoryview)):
        value_bytes = bytes(value)
        return _VALUE_TAG_BYTES + value_bytes, value_bytes

    encoded = _value_to_bytes(value)
    encoded = _as_bytes("serialized value", encoded)
    return _VALUE_TAG_SERIALIZED + encoded, _value_from_bytes(encoded)


def _deserialize_value(data: bytes):
    data = _as_bytes("value", data)

    if data.startswith(_VALUE_TAG_BYTES):
        return bytes(data[len(_VALUE_TAG_BYTES):])

    if data.startswith(_VALUE_TAG_SERIALIZED):
        return _value_from_bytes(data[len(_VALUE_TAG_SERIALIZED):])

    return bytes(data)


class KVStore:
    def __init__(self, path: str):
        self._wal_path = _resolve_wal_path(path)
        self._index = {}
        self._closed = False

        parent = os.path.dirname(os.path.abspath(self._wal_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._file = open(self._wal_path, "a+b")
        self._replay()
        self._file.seek(0, os.SEEK_END)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("KVStore is closed")

    def _replay(self) -> None:
        self._file.seek(0)
        data = self._file.read()

        offset = 0
        data_len = len(data)

        while offset < data_len:
            try:
                key, value, tombstone, next_offset = _decode_record(data, offset)
            except ValueError:
                break

            if next_offset <= offset:
                break

            if tombstone:
                self._index.pop(key, None)
            else:
                self._index[key] = _deserialize_value(value)

            offset = next_offset

        if offset != data_len:
            self._file.truncate(offset)
            _fsync_file(self._file)

    def _append_record(self, record: bytes) -> None:
        self._ensure_open()

        start = self._file.seek(0, os.SEEK_END)
        try:
            written = self._file.write(record)
            if written is not None and written != len(record):
                raise OSError("short WAL write")
            _fsync_file(self._file)
        except BaseException:
            try:
                self._file.truncate(start)
                _fsync_file(self._file)
                self._file.seek(0, os.SEEK_END)
            except Exception:
                pass
            raise

    def put(self, key: bytes, value):
        self._ensure_open()
        key = _as_bytes("key", key)

        stored_value, indexed_value = _serialize_value(value)
        record = _encode_record(key, stored_value, tombstone=False)
        self._append_record(record)
        self._index[key] = indexed_value

    def get(self, key: bytes):
        self._ensure_open()
        key = _as_bytes("key", key)
        return self._index.get(key)

    def delete(self, key: bytes):
        self._ensure_open()
        key = _as_bytes("key", key)

        record = _encode_record(key, b"", tombstone=True)
        self._append_record(record)
        self._index.pop(key, None)

    def keys(self) -> list:
        self._ensure_open()
        return list(self._index.keys())

    def close(self):
        if self._closed:
            return

        try:
            _fsync_file(self._file)
        finally:
            self._file.close()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass