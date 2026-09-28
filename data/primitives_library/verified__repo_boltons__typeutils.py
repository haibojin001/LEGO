import sys
from collections import deque


_issubclass = issubclass


def make_sentinel(name='_MISSING', var_name=None):
    """Create a unique falsey sentinel object.

    Args:
        name (str): Descriptive name for the sentinel.
        var_name (str): Module global name used to make the sentinel
            pickleable.
    """
    class Sentinel:
        def __init__(self):
            self.name = name
            self.var_name = var_name

        def __repr__(self):
            if self.var_name:
                return self.var_name
            return f'{self.__class__.__name__}({self.name!r})'

        def __bool__(self):
            return False

        def __copy__(self):
            return self

        def __deepcopy__(self, _memo):
            return self

        if var_name:
            def __reduce__(self):
                return self.var_name

    if var_name:
        frame = sys._getframe(1)
        module_name = frame.f_globals.get('__name__')
        if not module_name or module_name not in sys.modules:
            raise ValueError(
                'Pickleable sentinel objects (with var_name) can only'
                ' be created from top-level module scopes'
            )
        Sentinel.__module__ = module_name

    return Sentinel()


def issubclass(subclass, baseclass):
    """Safely test whether *subclass* inherits from *baseclass*.

    Unlike the builtin, invalid arguments return ``False`` instead of
    raising :class:`TypeError`.
    """
    try:
        return _issubclass(subclass, baseclass)
    except TypeError:
        return False


def get_all_subclasses(cls):
    """Return all direct and indirect subclasses of *cls* in breadth-first order."""
    try:
        pending = deque(cls.__subclasses__())
    except (AttributeError, TypeError):
        raise TypeError('expected type object, not %r' % cls)

    seen = set()
    subclasses = []
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        subclasses.append(current)
        pending.extend(current.__subclasses__())

    return subclasses


class classproperty:
    """A read-only property descriptor whose getter receives the class."""

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, cls):
        return self.fn(cls)