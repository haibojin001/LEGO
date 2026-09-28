"""CSV serialization utilities."""

from typing import Iterable, List, Sequence, Any

__all__ = ["dumps"]


def _validate_delimiter(delimiter: str) -> None:
    if not isinstance(delimiter, str):
        raise TypeError("delimiter must be a string")
    if delimiter == "":
        raise ValueError("delimiter must not be empty")
    if '"' in delimiter or "\r" in delimiter or "\n" in delimiter:
        raise ValueError("delimiter must not contain a quote or newline")


def _field_to_string(field: Any) -> str:
    if isinstance(field, str):
        return field
    return str(field)


def _needs_quotes(field: str, delimiter: str) -> bool:
    return (
        delimiter in field
        or '"' in field
        or "\n" in field
        or "\r" in field
    )


def _serialize_field(field: Any, delimiter: str) -> str:
    text = _field_to_string(field)
    if _needs_quotes(text, delimiter):
        return '"' + text.replace('"', '""') + '"'
    return text


def _serialize_row(row: Iterable[Any], delimiter: str) -> str:
    return delimiter.join(_serialize_field(field, delimiter) for field in row)


def dumps(rows, delimiter: str = ",") -> str:
    """Serialize rows to CSV using minimal RFC 4180-style quoting.

    A field is quoted only when it contains the delimiter, a double quote, or a
    newline. Embedded double quotes are escaped by doubling them. Rows are
    joined with CRLF and no trailing row separator is added.
    """
    _validate_delimiter(delimiter)
    return "\r\n".join(_serialize_row(row, delimiter) for row in rows)