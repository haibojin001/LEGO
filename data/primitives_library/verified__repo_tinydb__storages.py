import io
import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ('Storage', 'JSONStorage', 'MemoryStorage')


def touch(path: str, create_dirs: bool):
    """Ensure that *path* exists, optionally creating parent directories."""
    if create_dirs:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)

    with open(path, 'a'):
        pass


class Storage(ABC):
    """Abstract interface implemented by TinyDB storage backends."""

    @abstractmethod
    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        """Retrieve the stored database state, or ``None`` when empty."""
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def write(self, data: dict[str, dict[str, Any]]) -> None:
        """Persist the supplied database state."""
        raise NotImplementedError('To be overridden!')

    def close(self) -> None:
        """Release resources held by the storage, if any."""
        pass


class JSONStorage(Storage):
    """Storage backend that persists data in a JSON file."""

    def __init__(
        self,
        path: str,
        create_dirs=False,
        encoding=None,
        access_mode='r+',
        **kwargs
    ):
        super().__init__()

        self._mode = access_mode
        self.kwargs = kwargs

        if access_mode not in ('r', 'rb', 'r+', 'rb+'):
            warnings.warn(
                "Using an `access_mode` other than 'r', 'rb', 'r+' "
                "or 'rb+' can cause data loss or corruption"
            )

        if any(flag in access_mode for flag in ('+', 'w', 'a')):
            touch(path, create_dirs=create_dirs)

        self._handle = open(path, mode=access_mode, encoding=encoding)

    def close(self) -> None:
        self._handle.close()

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        self._handle.seek(0, os.SEEK_END)

        if self._handle.tell() == 0:
            return None

        self._handle.seek(0)
        return json.load(self._handle)

    def write(self, data: dict[str, dict[str, Any]]):
        self._handle.seek(0)
        payload = json.dumps(data, **self.kwargs)

        try:
            self._handle.write(payload)
        except io.UnsupportedOperation:
            raise IOError(
                'Cannot write to the database. Access mode is "{0}"'.format(
                    self._mode
                )
            )

        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.truncate()


class MemoryStorage(Storage):
    """Storage backend that retains the database state in memory."""

    def __init__(self):
        super().__init__()
        self.memory = None

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        return self.memory

    def write(self, data: dict[str, dict[str, Any]]):
        self.memory = data