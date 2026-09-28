import operator
from math import log as math_log
from itertools import chain, islice

try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
except ImportError:
    _MISSING = object()


__all__ = ['BList', 'BarrelList']


class BarrelList(list):
    _size_factor = 1520

    def __init__(self, iterable=None):
        self.lists = [[]]
        if iterable:
            self.extend(iterable)

    @property
    def _cur_size_limit(self):
        return int(round(self._size_factor * math_log(len(self) + 2, 2)))

    def _translate_index(self, index):
        if index < 0:
            index += len(self)

        relative_index = index
        for list_index, cur_list in enumerate(self.lists):
            cur_len = len(cur_list)
            if relative_index < cur_len:
                return list_index, relative_index
            relative_index -= cur_len

        return None, None

    def _balance_list(self, list_idx):
        if list_idx < 0:
            list_idx += len(self.lists)

        current = self.lists[list_idx]
        limit = self._cur_size_limit

        if len(current) > limit:
            half_limit = limit // 2
            while len(current) > half_limit:
                next_idx = list_idx + 1
                self.lists.insert(next_idx, current[-half_limit:])
                del current[-half_limit:]
            return True
        return False

    def _replace_all(self, values):
        self.lists[:] = [list(values)]
        self._balance_list(0)

    @classmethod
    def from_iterable(cls, it):
        return cls(it)

    def __bool__(self):
        return any(self.lists)

    __nonzero__ = __bool__

    def __len__(self):
        return sum(len(cur_list) for cur_list in self.lists)

    def __iter__(self):
        return chain.from_iterable(self.lists)

    def __reversed__(self):
        return chain.from_iterable(reversed(cur_list)
                                   for cur_list in reversed(self.lists))

    def __contains__(self, item):
        return any(item in cur_list for cur_list in self.lists)

    def insert(self, index, item):
        index = operator.index(index)
        total_len = len(self)

        if len(self.lists) == 1:
            self.lists[0].insert(index, item)
            self._balance_list(0)
            return

        if index >= total_len:
            list_idx = len(self.lists) - 1
            relative_idx = len(self.lists[-1])
        elif index < -total_len:
            list_idx = 0
            relative_idx = 0
        else:
            list_idx, relative_idx = self._translate_index(index)

        self.lists[list_idx].insert(relative_idx, item)
        self._balance_list(list_idx)

    def append(self, item):
        self.lists[-1].append(item)

    def extend(self, iterable):
        if iterable is self:
            iterable = list(self)
        self.lists[-1].extend(iterable)

    def pop(self, *args):
        lists = self.lists

        if len(lists) == 1 and not args:
            return lists[0].pop()

        index = args[0] if args else -1
        if index is None or index == () or index == -1:
            while len(lists) > 1 and not lists[-1]:
                lists.pop()
            result = lists[-1].pop()
            if len(lists) > 1 and not lists[-1]:
                lists.pop()
            return result

        index = operator.index(index)
        list_idx, relative_idx = self._translate_index(index)
        if list_idx is None:
            raise IndexError('pop index out of range')

        result = lists[list_idx].pop(relative_idx)
        if len(lists) > 1 and not lists[list_idx]:
            del lists[list_idx]
        else:
            self._balance_list(list_idx)
        return result

    def clear(self):
        self.lists[:] = [[]]

    def copy(self):
        return self.from_iterable(self)

    def __copy__(self):
        return self.copy()

    def iter_slice(self, start, stop, step=None):
        length = len(self)
        start, stop, step = slice(start, stop, step).indices(length)
        if step < 0:
            return islice(reversed(self),
                          length - 1 - start,
                          length - 1 - stop,
                          -step)
        return islice(self, start, stop, step)

    def del_slice(self, start, stop, step=None):
        values = list(self)
        del values[slice(start, stop, step)]
        self._replace_all(values)

    __delslice__ = del_slice

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.from_iterable(
                self.iter_slice(index.start, index.stop, index.step))

        index = operator.index(index)
        list_idx, relative_idx = self._translate_index(index)
        if list_idx is None:
            raise IndexError('list index out of range')
        return self.lists[list_idx][relative_idx]

    def __getslice__(self, start, stop):
        return self[start:stop]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            values = list(self)
            values[index] = value
            self._replace_all(values)
            return

        index = operator.index(index)
        list_idx, relative_idx = self._translate_index(index)
        if list_idx is None:
            raise IndexError('list assignment index out of range')
        self.lists[list_idx][relative_idx] = value

    def __setslice__(self, start, stop, values):
        self[start:stop] = values

    def __delitem__(self, index):
        if isinstance(index, slice):
            self.del_slice(index.start, index.stop, index.step)
            return

        index = operator.index(index)
        list_idx, relative_idx = self._translate_index(index)
        if list_idx is None:
            raise IndexError('list assignment index out of range')

        del self.lists[list_idx][relative_idx]
        if len(self.lists) > 1 and not self.lists[list_idx]:
            del self.lists[list_idx]
        elif self.lists:
            self._balance_list(list_idx)

    def __add__(self, other):
        if not isinstance(other, list):
            raise TypeError('can only concatenate list (not "%s") to list'
                            % type(other).__name__)
        return self.from_iterable(chain(self, other))

    def __radd__(self, other):
        if not isinstance(other, list):
            raise TypeError('can only concatenate list (not "%s") to list'
                            % type(other).__name__)
        return self.from_iterable(chain(other, self))

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __mul__(self, count):
        count = operator.index(count)
        return self.from_iterable(list(self) * count)

    def __rmul__(self, count):
        return self.__mul__(count)

    def __imul__(self, count):
        count = operator.index(count)
        self._replace_all(list(self) * count)
        return self

    def reverse(self):
        self._replace_all(reversed(self))

    def sort(self, *, key=None, reverse=False):
        values = list(self)
        values.sort(key=key, reverse=reverse)
        self._replace_all(values)

    def count(self, item):
        return sum(cur_list.count(item) for cur_list in self.lists)

    def index(self, item, start=0, stop=None):
        values = list(self)
        if stop is None:
            return values.index(item, start)
        return values.index(item, start, stop)

    def remove(self, item):
        for list_idx, cur_list in enumerate(self.lists):
            try:
                cur_list.remove(item)
            except ValueError:
                continue

            if len(self.lists) > 1 and not cur_list:
                del self.lists[list_idx]
            else:
                self._balance_list(list_idx)
            return
        raise ValueError('list.remove(x): x not in list')

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, list(self))

    def _comparison_value(self, other):
        if isinstance(other, BarrelList):
            return list(other)
        return other

    def __eq__(self, other):
        return list(self) == self._comparison_value(other)

    def __ne__(self, other):
        return list(self) != self._comparison_value(other)

    def __lt__(self, other):
        return list(self) < self._comparison_value(other)

    def __le__(self, other):
        return list(self) <= self._comparison_value(other)

    def __gt__(self, other):
        return list(self) > self._comparison_value(other)

    def __ge__(self, other):
        return list(self) >= self._comparison_value(other)


BList = BarrelList