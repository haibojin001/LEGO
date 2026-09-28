from __future__ import annotations

from add_trailing_comma._plugins import calls
from add_trailing_comma._plugins import classes
from add_trailing_comma._plugins import functions
from add_trailing_comma._plugins import imports
from add_trailing_comma._plugins import literals
from add_trailing_comma._plugins import match

PLUGINS = (
    calls,
    classes,
    functions,
    imports,
    literals,
    match,
)