from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

__all__ = ["natural_list"]


def natural_list(items: list[Any]) -> str:
    """Convert items into a naturally formatted list string."""
    count = len(items)
    if count == 0:
        return ""
    if count == 1:
        return str(items[0])
    if count == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(str(value) for value in items[:-1]) + f" and {items[-1]}"