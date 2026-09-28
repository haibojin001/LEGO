import os
import sys

_here = os.path.abspath(__file__)
_source = None

for _base in list(sys.path):
    if not _base:
        _base = os.getcwd()
    _candidate = os.path.abspath(os.path.join(_base, "yaml", "scanner.py"))
    if _candidate != _here and os.path.isfile(_candidate):
        _source = _candidate
        break

if _source is None:
    try:
        import site
        for _base in site.getsitepackages() + [site.getusersitepackages()]:
            _candidate = os.path.abspath(os.path.join(_base, "yaml", "scanner.py"))
            if _candidate != _here and os.path.isfile(_candidate):
                _source = _candidate
                break
    except Exception:
        pass

if _source is None:
    raise ImportError("unable to locate a compatible yaml.scanner implementation")

with open(_source, "rb") as _stream:
    exec(compile(_stream.read(), _source, "exec"), globals(), globals())

del os, sys, _here, _source, _base, _candidate, _stream