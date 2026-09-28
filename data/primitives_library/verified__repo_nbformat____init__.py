from __future__ import annotations

import os

from .convert import downgrade, upgrade
from .nbbase import (
    NotebookNode,
    new_author,
    new_code_cell,
    new_metadata,
    new_notebook,
    new_output,
    new_text_cell,
    new_worksheet,
)
from .nbjson import reads as read_json
from .nbjson import reads as reads_json
from .nbjson import to_notebook as to_notebook_json
from .nbjson import writes as write_json
from .nbjson import writes as writes_json
from .nbpy import reads as read_py
from .nbpy import reads as reads_py
from .nbpy import to_notebook as to_notebook_py
from .nbpy import writes as write_py
from .nbpy import writes as writes_py
from .nbxml import reads as read_xml
from .nbxml import reads as reads_xml
from .nbxml import to_notebook as to_notebook_xml

nbformat = 2
nbformat_minor = 0


def parse_filename(fname):
    """Return the normalized filename, notebook stem, and serialization type."""
    stem, suffix = os.path.splitext(fname)
    if suffix == ".ipynb" or suffix == ".json":
        kind = "json"
    elif suffix == ".py":
        kind = "py"
    else:
        stem = fname
        fname = fname + ".ipynb"
        kind = "json"
    return fname, stem, kind