from importlib import import_module as _import_module


def _publish(_module):
    _names = getattr(_module, "__all__", None)
    if _names is None:
        _names = (name for name in vars(_module) if not name.startswith("_"))
    globals().update({name: getattr(_module, name) for name in _names})


_publish(_import_module(".special", __package__))
_publish(_import_module(".slugify", __package__))

_version_info = _import_module(".__version__", __package__)

__title__ = _version_info.__title__
__author__ = _version_info.__author__
__author_email__ = _version_info.__author_email__
__description__ = _version_info.__description__
__url__ = _version_info.__url__
__license__ = _version_info.__license__
__copyright__ = _version_info.__copyright__
__version__ = _version_info.__version__

del _import_module, _publish, _version_info