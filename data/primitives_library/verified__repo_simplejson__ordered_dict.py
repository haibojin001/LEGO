"""Ordered dictionary implementation preserving insertion order."""

class DictMixin:
    """Small Python 3 compatibility implementation of the old UserDict mixin."""

    def has_key(self, key):
        return key in self

    def iterkeys(self):
        return iter(self)

    def itervalues(self):
        return (self[key] for key in self)

    def iteritems(self):
        return ((key, self[key]) for key in self)

    def keys(self):
        return list(self.iterkeys())

    def values(self):
        return list(self.itervalues())

    def items(self):
        return list(self.iteritems())

    def clear(self):
        for key in self.keys():
            del self[key]

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def pop(self, key, *args):
        if len(args) > 1:
            raise TypeError(
                "pop expected at most 2 arguments, got %d" % (len(args) + 1)
            )
        try:
            value = self[key]
        except KeyError:
            if args:
                return args[0]
            raise
        del self[key]
        return value

    def popitem(self):
        for key in self:
            value = self[key]
            del self[key]
            return key, value
        raise KeyError("container is empty")

    def update(self, other=None, **kwds):
        if other is not None:
            if hasattr(other, "iteritems"):
                for key, value in other.iteritems():
                    self[key] = value
            elif hasattr(other, "keys"):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwds.items():
            self[key] = value


class OrderedDict(dict, DictMixin):
    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError("expected at most 1 arguments, got %d" % len(args))
        try:
            self.__end
        except AttributeError:
            self.clear()
        self.update(*args, **kwds)

    def clear(self):
        end = []
        end[:] = [None, end, end]
        self.__end = end
        self.__map = {}
        dict.clear(self)

    def __setitem__(self, key, value):
        if key not in self:
            end = self.__end
            previous = end[1]
            node = [key, previous, end]
            previous[2] = node
            end[1] = node
            self.__map[key] = node
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        node = self.__map.pop(key)
        previous = node[1]
        following = node[2]
        previous[2] = following
        following[1] = previous

    def __iter__(self):
        end = self.__end
        node = end[2]
        while node is not end:
            yield node[0]
            node = node[2]

    def __reversed__(self):
        end = self.__end
        node = end[1]
        while node is not end:
            yield node[0]
            node = node[1]

    def popitem(self, last=True):
        if not self:
            raise KeyError("dictionary is empty")
        key = next(reversed(self)) if last else next(iter(self))
        value = self.pop(key)
        return key, value

    def __reduce__(self):
        entries = [[key, self[key]] for key in self]
        saved = self.__map, self.__end
        del self.__map
        del self.__end
        try:
            attributes = vars(self).copy()
        finally:
            self.__map, self.__end = saved
        if attributes:
            return self.__class__, (entries,), attributes
        return self.__class__, (entries,)

    def keys(self):
        return list(self)

    setdefault = DictMixin.setdefault
    update = DictMixin.update
    pop = DictMixin.pop
    values = DictMixin.values
    items = DictMixin.items
    iterkeys = DictMixin.iterkeys
    itervalues = DictMixin.itervalues
    iteritems = DictMixin.iteritems

    def __repr__(self):
        if not self:
            return "%s()" % self.__class__.__name__
        return "%s(%r)" % (self.__class__.__name__, self.items())

    def copy(self):
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        result = cls()
        for key in iterable:
            result[key] = value
        return result

    def __eq__(self, other):
        if isinstance(other, OrderedDict):
            return (
                len(self) == len(other)
                and all(left == right for left, right in zip(self.items(), other.items()))
            )
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other