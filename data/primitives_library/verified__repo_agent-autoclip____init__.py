"""Backend package."""

import os as _filesystem
from pkgutil import extend_path as _namespace_paths

__path__ = _namespace_paths(__path__, __name__)

_here = _filesystem.path.dirname(__file__)
_bases = (
    _here,
    _filesystem.path.dirname(_here),
    _filesystem.path.dirname(_filesystem.path.dirname(_here)),
)
_indicators = frozenset(
    ("utils", "repositories", "services", "models", "api", "core", "processing")
)
_ignored = frozenset(
    (
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "tests",
    )
)

for _base in _bases:
    if not _filesystem.path.isdir(_base):
        continue
    for _folder, _children, _names in _filesystem.walk(_base):
        _distance = _filesystem.path.relpath(_folder, _base)
        if _distance != "." and _distance.count(_filesystem.path.sep) >= 5:
            _children[:] = []
            continue
        _children[:] = [item for item in _children if item not in _ignored]
        _is_candidate = (
            bool(_indicators.intersection(_children))
            or "repositories.py" in _names
            or "processing_framework.py" in _names
            or "error_handler.py" in _names
        )
        if _is_candidate and _folder not in __path__:
            __path__.append(_folder)

del _base, _bases, _children, _distance, _filesystem, _folder, _here
del _ignored, _indicators, _is_candidate, _names, _namespace_paths