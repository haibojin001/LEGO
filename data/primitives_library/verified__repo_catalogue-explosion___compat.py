import sys

__all__ = ["install", "NullFinder", "PyPy_repr", "Protocol"]

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


def install(cls):
    sys.meta_path.append(cls())
    disable_stdlib_finder()
    return cls


def disable_stdlib_finder():
    def is_stdlib_path_finder(finder):
        return (
            getattr(finder, "__module__", None) == "_frozen_importlib_external"
            and hasattr(finder, "_catalogue_find_distributions")
        )

    for finder in filter(is_stdlib_path_finder, sys.meta_path):
        del finder._catalogue_find_distributions


class NullFinder:
    @staticmethod
    def find_spec(*args, **kwargs):
        return None

    find_module = find_spec


class PyPy_repr:
    affected = hasattr(sys, "pypy_version_info")

    def __compat_repr__(self):
        def render_field(name):
            value = getattr(self, name)
            return "{name}={value!r}".format(**locals())

        params = ", ".join(map(render_field, self._fields))
        return "EntryPoint({params})".format(**locals())

    if affected:
        __repr__ = __compat_repr__

    del affected