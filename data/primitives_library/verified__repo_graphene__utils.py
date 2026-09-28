import inspect
from functools import partial

from ..utils.module_loading import import_string
from .mountedtype import MountedType
from .unmountedtype import UnmountedType


def get_field_as(value, _as=None):
    if isinstance(value, MountedType):
        return value
    if isinstance(value, UnmountedType):
        if _as is None:
            return value
        return _as.mounted(value)
    return None


def yank_fields_from_attrs(attrs, _as=None, sort=True):
    collected = []

    for name, value in list(attrs.items()):
        field = get_field_as(value, _as)
        if field:
            collected.append((name, field))

    if sort:
        collected.sort(key=lambda item: item[1])

    return dict(collected)


def get_type(_type):
    if isinstance(_type, str):
        return import_string(_type)
    if inspect.isfunction(_type) or isinstance(_type, partial):
        return _type()
    return _type


def get_underlying_type(_type):
    while hasattr(_type, "of_type"):
        _type = _type.of_type
    return _type