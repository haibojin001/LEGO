from __future__ import annotations

from typing import Any

__all__ = ["Struct"]


class Struct(dict[Any, Any]):
    _allownew = True

    def __init__(self, *args, **kw):
        object.__setattr__(self, "_allownew", True)
        dict.__init__(self, *args, **kw)

    def __setitem__(self, key, value):
        if not self._allownew and key not in self:
            raise KeyError(
                "can't create new attribute %s when allow_new_attr(False)" % key
            )
        dict.__setitem__(self, key, value)

    def __setattr__(self, key, value):
        if isinstance(key, str):
            if key in self.__dict__ or hasattr(Struct, key):
                raise AttributeError(
                    "attr %s is a protected member of class Struct." % key
                )
        try:
            self.__setitem__(key, value)
        except KeyError as error:
            raise AttributeError(error) from None

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __iadd__(self, other):
        self.merge(other)
        return self

    def __add__(self, other):
        result = self.copy()
        result.merge(other)
        return result

    def __sub__(self, other):
        result = self.copy()
        result -= other
        return result

    def __isub__(self, other):
        for key in other:
            if key in self:
                del self[key]
        return self

    def __dict_invert(self, data):
        inverted = {}
        for key, values in data.items():
            if isinstance(values, str):
                values = values.split()
            for value in values:
                inverted[value] = key
        return inverted

    def dict(self):
        return self

    def copy(self):
        return Struct(dict.copy(self))

    def hasattr(self, key):
        return key in self

    def allow_new_attr(self, allow=True):
        object.__setattr__(self, "_allownew", allow)

    def merge(self, __loc_data__=None, __conflict_solve=None, **kw):
        if __loc_data__ is None:
            __loc_data__ = {}

        data = dict(__loc_data__, **kw)

        if __conflict_solve is None:
            __conflict_solve = {}

        conflict_solve = self.__dict_invert(__conflict_solve)

        policies = {
            "preserve": lambda old, new: old,
            "update": lambda old, new: new,
            "add": lambda old, new: old + new,
            "add_flip": lambda old, new: new + old,
            "add_s": lambda old, new: old + " " + new,
        }

        for key, value in data.items():
            if key not in self:
                self[key] = value
                continue

            try:
                solver = conflict_solve[key]
            except KeyError:
                continue

            if isinstance(solver, str):
                solver = policies[solver]

            self[key] = solver(self[key], value)