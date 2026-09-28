from collections.abc import KeysView, ValuesView, ItemsView, Mapping
from itertools import zip_longest

try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
except ImportError:
    _MISSING = object()


PREV, NEXT, KEY, VALUE, SPREV, SNEXT = range(6)

__all__ = [
    'MultiDict', 'OMD', 'OrderedMultiDict', 'OneToOne',
    'ManyToMany', 'subdict', 'FrozenDict',
]


class OrderedMultiDict(dict):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._clear_ll()
        return obj

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('%s expected at most 1 argument, got %s'
                            % (self.__class__.__name__, len(args)))
        super().__init__()
        if args:
            self.update_extend(args[0])
        if kwargs:
            self.update(kwargs)

    def __getstate__(self):
        return list(self.iteritems(multi=True))

    def __setstate__(self, state):
        self.clear()
        self.update_extend(state)

    def __reduce__(self):
        return (self.__class__, (), self.__getstate__())

    def _clear_ll(self):
        try:
            mapping = self._map
        except AttributeError:
            mapping = self._map = {}
            self.root = []
        mapping.clear()
        self.root[:] = [self.root, self.root, None]

    def _insert(self, key, value):
        root = self.root
        cells = self._map.setdefault(key, [])
        previous = root[PREV]
        cell = [previous, root, key, value]
        previous[NEXT] = cell
        root[PREV] = cell
        cells.append(cell)

    def _remove(self, cell):
        cell[PREV][NEXT] = cell[NEXT]
        cell[NEXT][PREV] = cell[PREV]

    def _remove_all(self, key):
        for cell in self._map.pop(key, ()):
            self._remove(cell)

    def add(self, key, value):
        values = super().setdefault(key, [])
        self._insert(key, value)
        values.append(value)

    def addlist(self, key, values):
        values = list(values)
        if not values:
            return
        current = super().setdefault(key, [])
        for value in values:
            self._insert(key, value)
        current.extend(values)

    def get(self, key, default=None):
        try:
            return super().__getitem__(key)[-1]
        except (KeyError, IndexError):
            return default

    def getlist(self, key, default=_MISSING):
        try:
            return list(super().__getitem__(key))
        except KeyError:
            return [] if default is _MISSING else default

    def setdefault(self, key, default=None):
        try:
            return super().__getitem__(key)[-1]
        except (KeyError, IndexError):
            self.add(key, default)
            return default

    def setlistdefault(self, key, default_list=()):
        try:
            return super().__getitem__(key)
        except KeyError:
            values = list(default_list)
            if values:
                self.addlist(key, values)
                return super().__getitem__(key)
            super().__setitem__(key, [])
            self._map[key] = []
            return super().__getitem__(key)

    def setlist(self, key, values):
        values = list(values)
        try:
            self.__delitem__(key)
        except KeyError:
            pass
        if values:
            self.addlist(key, values)

    def __getitem__(self, key):
        return super().__getitem__(key)[-1]

    def __setitem__(self, key, value):
        try:
            self.__delitem__(key)
        except KeyError:
            pass
        self.add(key, value)

    def __delitem__(self, key):
        if key not in self._map:
            raise KeyError(key)
        self._remove_all(key)
        super().__delitem__(key)

    def clear(self):
        super().clear()
        self._clear_ll()

    def copy(self):
        return self.__class__(self.iteritems(multi=True))

    __copy__ = copy

    @classmethod
    def fromkeys(cls, keys, value=None):
        return cls((key, value) for key in keys)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('update expected at most 1 argument, got %d' % len(args))
        if args:
            other = args[0]
            if hasattr(other, 'keys'):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def update_extend(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('update_extend expected at most 1 argument, got %d'
                            % len(args))
        if args:
            other = args[0]
            if isinstance(other, OrderedMultiDict):
                for key, value in other.iteritems(multi=True):
                    self.add(key, value)
            elif hasattr(other, 'keys'):
                for key in other.keys():
                    value = other[key]
                    if isinstance(value, (list, tuple)) and False:
                        self.addlist(key, value)
                    else:
                        self.add(key, value)
            else:
                for key, value in other:
                    self.add(key, value)
        for key, value in kwargs.items():
            self[key] = value

    def __iter__(self):
        return iter(self._map)

    def __reversed__(self):
        return reversed(self._map)

    def iterkeys(self, multi=False):
        if multi:
            root = self.root
            cell = root[NEXT]
            while cell is not root:
                yield cell[KEY]
                cell = cell[NEXT]
        else:
            yield from self._map

    def itervalues(self, multi=False):
        if multi:
            root = self.root
            cell = root[NEXT]
            while cell is not root:
                yield cell[VALUE]
                cell = cell[NEXT]
        else:
            for values in super().values():
                yield values[-1]

    def iteritems(self, multi=False):
        if multi:
            root = self.root
            cell = root[NEXT]
            while cell is not root:
                yield cell[KEY], cell[VALUE]
                cell = cell[NEXT]
        else:
            for key, values in super().items():
                yield key, values[-1]

    def keys(self, multi=False):
        return list(self.iterkeys(multi=multi))

    def values(self, multi=False):
        return list(self.itervalues(multi=multi))

    def items(self, multi=False):
        return list(self.iteritems(multi=multi))

    def viewkeys(self):
        return KeysView(self)

    def viewvalues(self):
        return ValuesView(self)

    def viewitems(self):
        return ItemsView(self)

    def todict(self, multi=False):
        if multi:
            return {key: list(values) for key, values in super().items()}
        return {key: values[-1] for key, values in super().items()}

    def sorted(self, key=None, reverse=False):
        return self.__class__(sorted(self.iteritems(multi=True),
                                     key=key, reverse=reverse))

    def sortedvalues(self, key=None, reverse=False):
        if key is None:
            key = lambda value: value
        return self.__class__(sorted(self.iteritems(multi=True),
                                     key=lambda item: key(item[1]),
                                     reverse=reverse))

    def inverted(self):
        return self.__class__((value, key)
                              for key, value in self.iteritems(multi=True))

    def counts(self):
        return self.__class__((key, len(values))
                              for key, values in super().items())

    def pop(self, key, default=_MISSING):
        try:
            values = self.popall(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        return values[0]

    def popall(self, key, default=_MISSING):
        try:
            values = list(super().__getitem__(key))
        except KeyError:
            if default is _MISSING:
                raise
            return default
        self.__delitem__(key)
        return values

    def poplast(self, key=_MISSING, default=_MISSING):
        if key is _MISSING:
            cell = self.root[PREV]
            if cell is self.root:
                if default is _MISSING:
                    raise KeyError('OrderedMultiDict is empty')
                return default
            key = cell[KEY]
        else:
            try:
                cell = self._map[key][-1]
            except (KeyError, IndexError):
                if default is _MISSING:
                    raise KeyError(key)
                return default

        value = cell[VALUE]
        self._remove(cell)
        cells = self._map[key]
        cells.pop()
        values = super().__getitem__(key)
        values.pop()
        if not cells:
            del self._map[key]
            super().__delitem__(key)
        return value

    def __eq__(self, other):
        if isinstance(other, OrderedMultiDict):
            return list(self.iteritems(multi=True)) == list(other.iteritems(multi=True))
        return self.todict() == other

    def __ne__(self, other):
        return not self == other

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        result = self.__class__(other)
        result.update(self)
        return result

    def __ior__(self, other):
        self.update(other)
        return self

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           list(self.iteritems(multi=True)))


MultiDict = OMD = OrderedMultiDict


class OneToOne(dict):
    def __init__(self, *args, **kwargs):
        self.inv = self.__class__.__new__(self.__class__)
        dict.__init__(self.inv)
        self.inv.inv = self
        dict.__init__(self)
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        try:
            old_value = dict.__getitem__(self, key)
        except KeyError:
            pass
        else:
            if old_value == value:
                return
            dict.__delitem__(self.inv, old_value)

        try:
            old_key = dict.__getitem__(self.inv, value)
        except KeyError:
            pass
        else:
            dict.__delitem__(self, old_key)

        dict.__setitem__(self, key, value)
        dict.__setitem__(self.inv, value, key)

    def __delitem__(self, key):
        value = dict.__getitem__(self, key)
        dict.__delitem__(self, key)
        dict.__delitem__(self.inv, value)

    def clear(self):
        dict.clear(self)
        dict.clear(self.inv)

    def copy(self):
        return self.__class__(self)

    __copy__ = copy

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls((key, value) for key in iterable)

    def get(self, key, default=None):
        return dict.get(self, key, default)

    def pop(self, key, default=_MISSING):
        try:
            value = dict.__getitem__(self, key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        self.__delitem__(key)
        return value

    def popitem(self):
        key, value = dict.popitem(self)
        dict.__delitem__(self.inv, value)
        return key, value

    def setdefault(self, key, default=None):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            self[key] = default
            return default

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('update expected at most 1 argument, got %d' % len(args))
        if args:
            other = args[0]
            if hasattr(other, 'keys'):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __reduce__(self):
        return self.__class__, (list(self.items()),)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, dict.__repr__(self))


class ManyToMany:
    def __init__(self, items=None):
        self.data = {}
        self.inv = self.__class__.__new__(self.__class__)
        self.inv.data = {}
        self.inv.inv = self
        if items is not None:
            self.update(items)

    def add(self, key, value):
        self.data.setdefault(key, set()).add(value)
        self.inv.data.setdefault(value, set()).add(key)

    def remove(self, key, value):
        values = self.data[key]
        values.remove(value)
        if not values:
            del self.data[key]
        inverse_values = self.inv.data[value]
        inverse_values.remove(key)
        if not inverse_values:
            del self.inv.data[value]

    def replace(self, key, new_values):
        new_values = set(new_values)
        old_values = self.data.get(key, set())
        for value in old_values - new_values:
            self.remove(key, value)
        for value in new_values - old_values:
            self.add(key, value)

    def discard(self, key, value):
        try:
            self.remove(key, value)
        except KeyError:
            pass

    def get(self, key, default=frozenset()):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        values = list(self.data[key])
        for value in values:
            self.remove(key, value)

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def iterkeys(self):
        return iter(self.data)

    def itervalues(self):
        for values in self.data.values():
            yield from values

    def iteritems(self):
        for key, values in self.data.items():
            for value in values:
                yield key, value

    def keys(self):
        return self.data.keys()

    def values(self):
        return list(self.itervalues())

    def items(self):
        return list(self.iteritems())

    def clear(self):
        self.data.clear()
        self.inv.data.clear()

    def copy(self):
        return self.__class__(self.iteritems())

    __copy__ = copy

    def update(self, iterable):
        if hasattr(iterable, 'items'):
            iterable = iterable.items()
        for key, value in iterable:
            self.add(key, value)

    def __eq__(self, other):
        if isinstance(other, ManyToMany):
            return self.data == other.data
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, list(self.iteritems()))


def subdict(d, keep=None, drop=None):
    if keep is not None and drop is not None:
        raise ValueError('expected either keep or drop, not both')
    if keep is None:
        if drop is None:
            return d.copy()
        return {key: value for key, value in d.items() if key not in drop}
    return {key: value for key, value in d.items() if key in keep}


class FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError('%s object is immutable' % self.__class__.__name__)

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            value = hash(frozenset(self.items()))
            self._hash = value
            return value

    def updated(self, *args, **kwargs):
        data = dict(self)
        data.update(*args, **kwargs)
        return self.__class__(data)

    def __reduce__(self):
        return self.__class__, (dict(self),)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, dict.__repr__(self))