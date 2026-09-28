"""Utilities for inspecting and temporarily controlling Python's garbage collector."""

import gc as _gc
import sys as _sys

__all__ = ['get_all', 'GCToggler', 'toggle_gc', 'toggle_gc_postcollect']


def get_all(type_obj, include_subtypes=True):
    """Return tracked instances of *type_obj* currently known to the GC."""
    if not isinstance(type_obj, type):
        raise TypeError('expected a type, not %r' % type_obj)

    try:
        class_is_tracked = _gc.is_tracked(type_obj)
    except AttributeError:
        class_is_tracked = False

    candidates = (_gc.get_referrers(type_obj)
                  if class_is_tracked else _gc.get_objects())

    if include_subtypes:
        return [candidate for candidate in candidates
                if isinstance(candidate, type_obj)]
    return [candidate for candidate in candidates
            if type(candidate) is type_obj]


if '__pypy__' in _sys.builtin_module_names:
    del get_all


class GCToggler:
    """Context manager which disables cyclic garbage collection temporarily."""

    def __init__(self, postcollect=False):
        self.postcollect = postcollect

    def __enter__(self):
        _gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        _gc.enable()
        if self.postcollect:
            _gc.collect()


toggle_gc = GCToggler()
toggle_gc_postcollect = GCToggler(postcollect=True)