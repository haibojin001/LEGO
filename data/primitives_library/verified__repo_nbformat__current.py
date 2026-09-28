from __future__ import annotations

import re
import warnings

from traitlets.log import get_logger

from nbformat import v3 as _v_latest
from nbformat.v3 import (
    NotebookNode,
    nbformat,
    nbformat_minor,
    nbformat_schema,
    new_author,
    new_code_cell,
    new_heading_cell,
    new_metadata,
    new_notebook,
    new_output,
    new_text_cell,
    new_worksheet,
    parse_filename,
    to_notebook_json,
)

from . import versions
from .converter import convert
from .reader import reads as reader_reads
from .validator import ValidationError, _validate, validate

warnings.warn(
    """nbformat.current is deprecated since before nbformat 3.0

- use nbformat for read/write/validate public API
- use nbformat.vX directly to composing notebooks of a particular version
""",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "NBFormatError",
    "NotebookNode",
    "convert",
    "nbformat",
    "nbformat_minor",
    "nbformat_schema",
    "new_author",
    "new_code_cell",
    "new_heading_cell",
    "new_metadata",
    "new_notebook",
    "new_output",
    "new_text_cell",
    "new_worksheet",
    "parse_filename",
    "parse_py",
    "read",
    "reads",
    "reads_json",
    "reads_py",
    "to_notebook_json",
    "validate",
    "write",
    "writes",
    "writes_json",
    "writes_py",
]

current_nbformat = nbformat
current_nbformat_minor = nbformat_minor
current_nbformat_module = _v_latest.__name__


class NBFormatError(ValueError):
    """An error raised for an nbformat error."""


def _warn_format():
    warnings.warn(
        """Non-JSON file support in nbformat is deprecated since nbformat 1.0.
    Use nbconvert to create files of other formats.""",
        stacklevel=2,
    )


def parse_py(s, **kwargs):
    """Parse a string into a (nbformat, string) tuple."""
    major = current_nbformat
    minor = current_nbformat_minor
    match = re.search(r"# <nbformat>(?P<nbformat>\d+[\.\d+]*)</nbformat>", s)
    if match is not None:
        parts = match.group("nbformat").split(".")
        major = int(parts[0])
        if len(parts) > 1:
            minor = int(parts[1])
    return major, minor, s


def reads_json(nbjson, **kwargs):
    """DEPRECATED, use reads."""
    warnings.warn(
        "reads_json is deprecated since nbformat 3.0, use reads",
        DeprecationWarning,
        stacklevel=2,
    )
    return reads(nbjson)


def writes_json(nb, **kwargs):
    """DEPRECATED, use writes."""
    warnings.warn(
        "writes_json is deprecated since nbformat 3.0, use writes",
        DeprecationWarning,
        stacklevel=2,
    )
    return writes(nb, **kwargs)


def reads_py(s, **kwargs):
    """DEPRECATED: use nbconvert."""
    _warn_format()
    version, _minor, source = parse_py(s, **kwargs)
    if version in (2, 3):
        return versions[version].to_notebook_py(source, **kwargs)
    raise NBFormatError("Unsupported PY nbformat version: %i" % version)


def writes_py(nb, **kwargs):
    """DEPRECATED: use nbconvert."""
    _warn_format()
    return versions[3].writes_py(nb, **kwargs)


def reads(s, format="DEPRECATED", version=current_nbformat, **kwargs):
    """Read a notebook from a string and return the NotebookNode object."""
    if format not in {"DEPRECATED", "json"}:
        _warn_format()
    notebook = reader_reads(s, **kwargs)
    notebook = convert(notebook, version)
    try:
        _validate(notebook, repair_duplicate_cell_ids=False)
    except ValidationError as error:
        get_logger().error("Notebook JSON is invalid: %s", error)
    return notebook


def writes(nb, format="DEPRECATED", version=current_nbformat, **kwargs):
    """Write a notebook to a string in a given format in the current nbformat version."""
    if format not in {"DEPRECATED", "json"}:
        _warn_format()
    notebook = convert(nb, version)
    try:
        _validate(notebook, repair_duplicate_cell_ids=False)
    except ValidationError as error:
        get_logger().error("Notebook JSON is invalid: %s", error)
    return versions[version].writes_json(notebook, **kwargs)


def read(fp, format="DEPRECATED", **kwargs):
    """Read a notebook from a file and return the NotebookNode object."""
    return reads(fp.read(), **kwargs)


def write(nb, fp, format="DEPRECATED", **kwargs):
    """Write a notebook to a file in a given format in the current nbformat version."""
    data = writes(nb, **kwargs)
    if isinstance(data, bytes):
        data = data.decode("utf8")
    return fp.write(data)