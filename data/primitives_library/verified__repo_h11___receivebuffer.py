import re
import sys
from typing import List, Optional, Union

__all__ = ["ReceiveBuffer"]

blank_line_regex = re.compile(b"\n\r?\n", re.MULTILINE)


class ReceiveBuffer:
    def __init__(self) -> None:
        self._data = bytearray()
        self._next_line_search = 0
        self._multiple_lines_search = 0

    def __iadd__(self, byteslike: Union[bytes, bytearray]) -> "ReceiveBuffer":
        self._data.extend(byteslike)
        return self

    def __bool__(self) -> bool:
        return bool(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __bytes__(self) -> bytes:
        return bytes(self._data)

    def _extract(self, count: int) -> bytearray:
        extracted = self._data[:count]
        del self._data[:count]
        self._next_line_search = 0
        self._multiple_lines_search = 0
        return extracted

    def maybe_extract_at_most(self, count: int) -> Optional[bytearray]:
        if not self._data[:count]:
            return None
        return self._extract(count)

    def maybe_extract_next_line(self) -> Optional[bytearray]:
        start = max(self._next_line_search - 1, 0)
        position = self._data.find(b"\r\n", start)

        if position < 0:
            self._next_line_search = len(self._data)
            return None

        return self._extract(position + 2)

    def maybe_extract_lines(self) -> Optional[List[bytearray]]:
        if self._data[:1] == b"\n":
            self._extract(1)
            return []

        if self._data[:2] == b"\r\n":
            self._extract(2)
            return []

        found = blank_line_regex.search(self._data, self._multiple_lines_search)
        if found is None:
            self._multiple_lines_search = max(len(self._data) - 2, 0)
            return None

        extracted = self._extract(found.end())
        lines = extracted.split(b"\n")

        for line in lines:
            if line.endswith(b"\r"):
                del line[-1]

        assert lines[-2] == b""
        assert lines[-1] == b""
        del lines[-2:]
        return lines

    def is_next_line_obviously_invalid_request_line(self) -> bool:
        try:
            return self._data[0] < 0x21
        except IndexError:
            return False