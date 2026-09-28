import re
from typing import AnyStr, List, Sequence, Tuple, TYPE_CHECKING, Union, cast, overload

from ._abnf import field_name, field_value
from ._util import LocalProtocolError, bytesify, validate

if TYPE_CHECKING:
    from ._events import Request

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


CONTENT_LENGTH_MAX_DIGITS = 20

_content_length_re = re.compile(rb"[0-9]+")
_field_name_re = re.compile(field_name.encode("ascii"))
_field_value_re = re.compile(field_value.encode("ascii"))


class Headers(Sequence[Tuple[bytes, bytes]]):
    __slots__ = ("_full_items",)

    def __init__(self, full_items: List[Tuple[bytes, bytes, bytes]]) -> None:
        self._full_items = full_items

    def __bool__(self) -> bool:
        return len(self._full_items) != 0

    def __eq__(self, other: object) -> bool:
        return list(self) == list(other)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._full_items)

    def __repr__(self) -> str:
        return "<Headers({})>".format(repr(list(self)))

    def __getitem__(self, idx: int) -> Tuple[bytes, bytes]:  # type: ignore[override]
        item = self._full_items[idx]
        return (item[1], item[2])

    def raw_items(self) -> List[Tuple[bytes, bytes]]:
        return [(item[0], item[2]) for item in self._full_items]


HeaderTypes = Union[
    List[Tuple[bytes, bytes]],
    List[Tuple[bytes, str]],
    List[Tuple[str, bytes]],
    List[Tuple[str, str]],
]


@overload
def normalize_and_validate(headers: Headers, _parsed: Literal[True]) -> Headers:
    ...


@overload
def normalize_and_validate(headers: HeaderTypes, _parsed: Literal[False]) -> Headers:
    ...


@overload
def normalize_and_validate(
    headers: Union[Headers, HeaderTypes], _parsed: bool = False
) -> Headers:
    ...


def normalize_and_validate(
    headers: Union[Headers, HeaderTypes], _parsed: bool = False
) -> Headers:
    output: List[Tuple[bytes, bytes, bytes]] = []
    content_length: Union[bytes, None] = None
    transfer_encoding_seen = False

    for name, value in headers:
        if not _parsed:
            name = bytesify(name)
            value = bytesify(value)
            validate(_field_name_re, name, "Illegal header name {!r}", name)
            validate(_field_value_re, value, "Illegal header value {!r}", value)

        assert isinstance(name, bytes)
        assert isinstance(value, bytes)

        original_name = name
        normalized_name = name.lower()

        if normalized_name == b"content-length":
            declared_lengths = {part.strip() for part in value.split(b",")}
            if len(declared_lengths) != 1:
                raise LocalProtocolError("conflicting Content-Length headers")

            normalized_value = declared_lengths.pop()
            validate(_content_length_re, normalized_value, "bad Content-Length")

            if len(normalized_value) > CONTENT_LENGTH_MAX_DIGITS:
                raise LocalProtocolError("bad Content-Length")

            if content_length is None:
                content_length = normalized_value
                output.append((original_name, normalized_name, normalized_value))
            elif content_length != normalized_value:
                raise LocalProtocolError("conflicting Content-Length headers")

        elif normalized_name == b"transfer-encoding":
            if transfer_encoding_seen:
                raise LocalProtocolError(
                    "multiple Transfer-Encoding headers", error_status_hint=501
                )

            normalized_value = value.lower()
            if normalized_value != b"chunked":
                raise LocalProtocolError(
                    "Only Transfer-Encoding: chunked is supported",
                    error_status_hint=501,
                )

            transfer_encoding_seen = True
            output.append((original_name, normalized_name, normalized_value))

        else:
            output.append((original_name, normalized_name, value))

    return Headers(output)


def get_comma_header(headers: Headers, name: bytes) -> List[bytes]:
    values: List[bytes] = []

    for _, found_name, raw_value in headers._full_items:
        if found_name != name:
            continue

        for piece in raw_value.lower().split(b","):
            piece = piece.strip()
            if piece:
                values.append(piece)

    return values


def set_comma_header(headers: Headers, name: bytes, new_values: List[bytes]) -> Headers:
    replacement = b", ".join(new_values)
    items: List[Tuple[bytes, bytes, bytes]] = []

    for raw_name, normalized_name, value in headers._full_items:
        if normalized_name == name:
            items.append((raw_name, normalized_name, replacement))
        else:
            items.append((raw_name, normalized_name, value))

    return Headers(items)


def has_expect_100_continue(request: "Request") -> bool:
    if request.http_version < b"1.1":
        return False
    return b"100-continue" in get_comma_header(request.headers, b"expect")