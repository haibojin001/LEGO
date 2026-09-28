from __future__ import annotations

import copy
import json

from .nbbase import from_dict
from .rwbase import NotebookReader, NotebookWriter, rejoin_lines, restore_bytes, split_lines


class BytesEncoder(json.JSONEncoder):
    """JSON encoder supporting ASCII byte strings."""

    def default(self, obj):
        """Return a JSON-compatible representation for unsupported objects."""
        if isinstance(obj, bytes):
            return obj.decode("ascii")
        return super().default(obj)


class JSONReader(NotebookReader):
    """Reader for version 2 notebooks stored as JSON."""

    def reads(self, s, **kwargs):
        """Read notebook JSON from a string."""
        data = json.loads(s, **kwargs)
        return self.to_notebook(data, **kwargs)

    def to_notebook(self, d, **kwargs):
        """Convert a decoded JSON dictionary into a notebook object."""
        return restore_bytes(rejoin_lines(from_dict(d)))


class JSONWriter(NotebookWriter):
    """Writer for version 2 notebooks stored as JSON."""

    def writes(self, nb, **kwargs):
        """Serialize a notebook object as JSON."""
        kwargs["cls"] = BytesEncoder
        kwargs["indent"] = 1
        kwargs["sort_keys"] = True

        if kwargs.pop("split_lines", True):
            nb = split_lines(copy.deepcopy(nb))

        return json.dumps(nb, **kwargs)


_reader = JSONReader()
_writer = JSONWriter()

reads = _reader.reads
read = _reader.read
to_notebook = _reader.to_notebook
write = _writer.write
writes = _writer.writes