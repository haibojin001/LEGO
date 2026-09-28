from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from w3lib.html import replace_entities as w3lib_replace_entities

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def flatten(x: Iterable[Any]) -> list[Any]:
    """Return all recursively nested iterable items as a flat list."""
    return list(iflatten(x))


def iflatten(x: Iterable[Any]) -> Iterator[Any]:
    """Yield all recursively nested iterable items in order."""
    for item in x:
        if _is_listlike(item):
            yield from flatten(item)
        else:
            yield item


def _is_listlike(x: Any) -> bool:
    """Return whether *x* is iterable but not text or bytes."""
    return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))


def extract_regex(
    regex: str | re.Pattern[str], text: str, replace_entities: bool = True
) -> list[str]:
    """Extract strings from text according to regular-expression groups."""
    if isinstance(regex, str):
        regex = re.compile(regex, re.UNICODE)

    if "extract" in regex.groupindex:
        try:
            value = cast("re.Match[str]", regex.search(text)).group("extract")
        except AttributeError:
            values: list[Any] = []
        else:
            values = [value] if value is not None else []
    else:
        values = regex.findall(text)

    result = flatten(values)
    if not replace_entities:
        return result
    return [w3lib_replace_entities(value, keep=["lt", "amp"]) for value in result]


def shorten(text: str, width: int, suffix: str = "...") -> str:
    """Truncate text to the requested width, appending a suffix when possible."""
    if len(text) <= width:
        return text
    if width > len(suffix):
        return text[: width - len(suffix)] + suffix
    if width >= 0:
        return suffix[len(suffix) - width :]
    raise ValueError("width must be equal or greater than 0")