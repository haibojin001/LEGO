from __future__ import annotations

import json

from .validator import ValidationError


class NotJSONError(ValueError):
    """Raised when notebook content cannot be decoded as JSON."""


def parse_json(s, **kwargs):
    """Decode JSON notebook content."""
    try:
        return json.loads(s, **kwargs)
    except ValueError as error:
        text = f"Notebook does not appear to be JSON: {s!r}"
        if len(text) > 80:
            text = text[:77] + "..."
        raise NotJSONError(text) from error


def get_version(nb):
    """Return the notebook major and minor format versions."""
    return nb.get("nbformat", 1), nb.get("nbformat_minor", 0)


def reads(s, **kwargs):
    """Read notebook JSON content into the corresponding notebook object."""
    from . import NBFormatError, versions

    notebook_data = parse_json(s, **kwargs)
    major, minor = get_version(notebook_data)

    if major not in versions:
        raise NBFormatError("Unsupported nbformat version %s" % major)

    try:
        return versions[major].to_notebook_json(notebook_data, minor=minor)
    except AttributeError as error:
        raise ValidationError(
            f"The notebook is invalid and is missing an expected key: {error}"
        ) from None


def read(fp, **kwargs):
    """Read a notebook from a file-like object's contents."""
    return reads(fp.read(), **kwargs)