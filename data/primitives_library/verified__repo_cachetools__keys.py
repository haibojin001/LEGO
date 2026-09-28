"""Key functions for memoizing decorators."""

__all__ = ("hashkey", "methodkey", "typedkey", "typedmethodkey")


class _HashedTuple(tuple):
    """Tuple subclass which remembers its computed hash value."""

    __hashvalue = None

    def __hash__(self, hash=tuple.__hash__):
        value = self.__hashvalue
        if value is None:
            value = hash(self)
            self.__hashvalue = value
        return value

    def __add__(self, other, add=tuple.__add__):
        return _HashedTuple(add(self, other))

    def __radd__(self, other, add=tuple.__add__):
        return _HashedTuple(add(other, self))

    def __getstate__(self):
        return {}


_kwmark = (_HashedTuple,)


def hashkey(*args, **kwargs):
    """Return a cache key for the specified hashable arguments."""
    if kwargs:
        items = tuple(sorted(kwargs.items()))
        return _HashedTuple(args + _kwmark + items)
    return _HashedTuple(args)


def methodkey(self, *args, **kwargs):
    """Return a cache key for use with cached methods."""
    return hashkey(*args, **kwargs)


def typedkey(*args, **kwargs):
    """Return a cache key for the specified hashable arguments and types."""
    if kwargs:
        items = tuple(sorted(kwargs.items()))
        key = _HashedTuple(args + _kwmark + items)
        key += tuple(type(value) for _, value in items)
    else:
        key = _HashedTuple(args)
    return key + tuple(type(value) for value in args)


def typedmethodkey(self, *args, **kwargs):
    """Return a typed cache key for use with cached methods."""
    return typedkey(*args, **kwargs)