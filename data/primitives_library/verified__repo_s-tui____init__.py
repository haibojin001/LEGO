"""Convenience access to the widgets and helpers contained in this package."""

import importlib as _importlib
import importlib.util as _importlib_util
import pkgutil as _pkgutil
import re as _re
import threading as _threading

_lock = _threading.RLock()
_modules = None
_lookup = {}


def _module_names():
    global _modules
    with _lock:
        if _modules is None:
            _modules = tuple(
                name
                for _, name, _ in _pkgutil.iter_modules(__path__)
                if not name.startswith("_")
            )
        return _modules


def _snake_case(name):
    return _re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _load_module(name):
    module = _importlib.import_module("%s.%s" % (__name__, name))
    globals().setdefault(name, module)
    return module


def _find_in_modules(name):
    cached = _lookup.get(name)
    if cached is not None:
        return cached

    names = _module_names()
    candidates = []
    for candidate in (name, name.lower(), _snake_case(name)):
        if candidate in names and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        module = _load_module(candidate)
        if hasattr(module, name):
            value = getattr(module, name)
            _lookup[name] = value
            globals()[name] = value
            return value
        if candidate == name:
            _lookup[name] = module
            globals()[name] = module
            return module

    for module_name in names:
        if module_name in candidates:
            continue
        try:
            module = _load_module(module_name)
        except Exception:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            _lookup[name] = value
            globals()[name] = value
            return value

    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    return _find_in_modules(name)


def __dir__():
    names = set(globals())
    names.update(_module_names())
    for module_name in _module_names():
        try:
            module = _load_module(module_name)
        except Exception:
            continue
        names.update(name for name in vars(module) if not name.startswith("_"))
    return sorted(names)