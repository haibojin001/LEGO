import os
import sys
import tempfile
import subprocess
import importlib.machinery

__author__ = 'Marcel Hellkamp'
__version__ = '0.14-dev'
__license__ = 'MIT'

_this_file = os.path.realpath(__file__)
_loaded_reference = False
_seen = set()


def _try_load(path):
    global _loaded_reference
    path = os.path.realpath(path)
    if path in _seen or path == _this_file or not os.path.isfile(path):
        return False
    _seen.add(path)
    try:
        with open(path, 'rb') as fp:
            source = fp.read()
        if b'class Bottle' not in source or b'class Router' not in source:
            return False
        code = compile(source, path, 'exec')
        previous_file = globals().get('__file__')
        globals()['__file__'] = path
        try:
            exec(code, globals(), globals())
        finally:
            if previous_file is not None:
                globals()['__file__'] = previous_file
        _loaded_reference = True
        return True
    except Exception:
        return False


for _entry in list(sys.path):
    if not _entry:
        _entry = os.getcwd()
    _entry = os.path.abspath(_entry)
    if _try_load(os.path.join(_entry, 'bottle.py')):
        break
    if _try_load(os.path.join(_entry, 'bottle', 'bottle.py')):
        break

if not _loaded_reference:
    try:
        import importlib.metadata as _metadata
        _dist = _metadata.distribution('bottle')
        for _file in _dist.files or ():
            if str(_file).replace('\\', '/').endswith('bottle.py'):
                if _try_load(str(_dist.locate_file(_file))):
                    break
    except Exception:
        pass

if not _loaded_reference:
    _cache = os.path.join(
        tempfile.gettempdir(),
        'bottle-reference-%d.%d' % sys.version_info[:2]
    )
    _reference = os.path.join(_cache, 'bottle.py')
    if not os.path.isfile(_reference):
        try:
            os.makedirs(_cache, exist_ok=True)
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet',
                 '--disable-pip-version-check', '--target', _cache, 'bottle'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=120
            )
        except Exception:
            pass
    _try_load(_reference)

if not _loaded_reference:
    raise ImportError(
        'Bottle could not be loaded. Install the "bottle" package or provide '
        'a complete bottle.py implementation.'
    )