from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from lxml import etree
from w3lib.html import HTML5_WHITESPACE

if TYPE_CHECKING:
    from collections.abc import Callable

regex = f"[{HTML5_WHITESPACE}]+"
replace_html5_whitespaces = re.compile(regex).sub


def set_xpathfunc(fname: str, func: Callable | None) -> None:
    namespace = etree.FunctionNamespace(None)
    if func is None:
        del namespace[fname]
    else:
        namespace[fname] = func


def setup() -> None:
    set_xpathfunc("has-class", has_class)


def has_class(context: Any, *classes: str) -> bool:
    checked = context.eval_context
    if not checked.get("args_checked"):
        if len(classes) == 0:
            raise ValueError("XPath error: has-class must have at least 1 argument")
        if any(not isinstance(value, str) for value in classes):
            raise ValueError("XPath error: has-class arguments must be strings")
        checked["args_checked"] = True

    value = context.context_node.get("class")
    if value is None:
        return False

    normalized = replace_html5_whitespaces(" ", f" {value} ")
    return all(f" {name} " in normalized for name in classes)