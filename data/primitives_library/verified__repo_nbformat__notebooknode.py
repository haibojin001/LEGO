from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy as _deepcopy

from ._struct import Struct

_ATOMIC_TYPES = (str, int, float, bool, bytes, type(None), complex)


class NotebookNode(Struct):
    """Dictionary-like notebook object supporting attribute access."""

    def __setitem__(self, key, value):
        if isinstance(value, Mapping) and not isinstance(value, NotebookNode):
            value = from_dict(value)
        super().__setitem__(key, value)

    def __deepcopy__(self, memo):
        copied = self.__class__()
        memo[id(self)] = copied

        for key, value in self.items():
            kind = type(value)
            if kind in _ATOMIC_TYPES:
                dict.__setitem__(copied, key, value)
            elif kind is list:
                dict.__setitem__(
                    copied,
                    key,
                    [_deepcopy(element, memo) for element in value],
                )
            else:
                copied[key] = _deepcopy(value, memo)

        if self.__dict__:
            copied.__dict__.update(_deepcopy(self.__dict__, memo))

        return copied

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError("update expected at most 1 arguments, got %d" % len(args))

        if args:
            source = args[0]
            if isinstance(source, Mapping):
                for key in source:
                    self[key] = source[key]
            elif hasattr(source, "keys"):
                for key in source:
                    self[key] = source[key]
            else:
                for key, value in source:
                    self[key] = value

        for key, value in kwargs.items():
            self[key] = value


def from_dict(d):
    """Recursively turn dictionaries in a container into NotebookNodes."""
    if isinstance(d, dict):
        return NotebookNode({key: from_dict(value) for key, value in d.items()})
    if isinstance(d, (tuple, list)):
        return [from_dict(value) for value in d]
    return d