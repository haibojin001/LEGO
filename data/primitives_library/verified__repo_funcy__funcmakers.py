from collections.abc import Mapping, Set
from operator import itemgetter

from .strings import _re_type, re_finder, re_tester

__all__ = ("make_func", "make_pred")


def make_func(f, test=False):
    if callable(f):
        return f

    if f is None:
        if test:
            return bool
        return lambda value: value

    if isinstance(f, (str, bytes, _re_type)):
        if test:
            return re_tester(f)
        return re_finder(f)

    if isinstance(f, (int, slice)):
        return itemgetter(f)

    if isinstance(f, Mapping):
        return f.__getitem__

    if isinstance(f, Set):
        return f.__contains__

    raise TypeError("Can't make a func from %s" % type(f).__name__)


def make_pred(pred):
    return make_func(pred, test=True)