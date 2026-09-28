from bisect import bisect_left
from collections.abc import MutableSet
from itertools import chain, islice
import operator

try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
except ImportError:
    _MISSING = object()


__all__ = ['IndexedSet', 'complement']


_COMPACTION_FACTOR = 8
_MAX_DEAD_INTERVALS = 384


class IndexedSet(MutableSet):
    def __init__(self, other=None):
        self.item_index_map = {}
        self.item_list = []
        self.dead_indices = []
        self._compactions = 0
        self._c_max_size = 0
        if other:
            self.update(other)

    @property
    def _dead_index_count(self):
        return len(self.item_list) - len(self.item_index_map)

    def _compact(self):
        if not self.dead_indices:
            return
        self._compactions += 1
        self._c_max_size = max(self._c_max_size, len(self.item_list))
        live_items = [item for item in self.item_list
                      if item is not _MISSING]
        self.item_list[:] = live_items
        self.item_index_map.clear()
        self.item_index_map.update(
            (item, index) for index, item in enumerate(live_items))
        del self.dead_indices[:]

    def _cull(self):
        if not self.dead_indices:
            return

        if not self.item_index_map:
            del self.item_list[:]
            del self.dead_indices[:]
            return

        if (len(self.dead_indices) > _MAX_DEAD_INTERVALS
                or self._dead_index_count > len(self.item_list) / _COMPACTION_FACTOR):
            self._compact()
            return

        if self.item_list and self.item_list[-1] is _MISSING:
            end = len(self.item_list)
            while end and self.item_list[end - 1] is _MISSING:
                end -= 1
            del self.item_list[end:]
            retained = []
            for start, stop in self.dead_indices:
                if start < end:
                    retained.append([start, min(stop, end)])
            self.dead_indices[:] = retained

    def _get_real_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('IndexedSet index out of range')

        real_index = index
        for start, stop in self.dead_indices:
            if real_index < start:
                break
            real_index += stop - start
        return real_index

    def _get_apparent_index(self, index):
        if index < 0:
            index += len(self)

        apparent_index = index
        for start, stop in self.dead_indices:
            if index < start:
                break
            apparent_index -= stop - start
        return apparent_index

    def _add_dead(self, start, stop=None):
        if stop is None:
            stop = start + 1

        intervals = self.dead_indices
        position = bisect_left(intervals, [start, stop])

        if position:
            prev_start, prev_stop = intervals[position - 1]
            if prev_stop >= start:
                start = min(start, prev_start)
                stop = max(stop, prev_stop)
                position -= 1
                del intervals[position]

        while position < len(intervals) and intervals[position][0] <= stop:
            stop = max(stop, intervals[position][1])
            del intervals[position]

        intervals.insert(position, [start, stop])

    def _bulk_discard(self, to_remove):
        self.item_list = [item for item in self.item_list
                          if item is not _MISSING and item not in to_remove]
        self.item_index_map = {
            item: index for index, item in enumerate(self.item_list)
        }
        del self.dead_indices[:]

    def __len__(self):
        return len(self.item_index_map)

    def __contains__(self, item):
        return item in self.item_index_map

    def __iter__(self):
        return (item for item in self.item_list if item is not _MISSING)

    def __reversed__(self):
        return (item for item in reversed(self.item_list)
                if item is not _MISSING)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, IndexedSet):
            return len(self) == len(other) and list(self) == list(other)
        try:
            return set(self) == set(other)
        except TypeError:
            return False

    __hash__ = None

    @classmethod
    def from_iterable(cls, it):
        return cls(it)

    def add(self, item):
        if item not in self.item_index_map:
            self.item_index_map[item] = len(self.item_list)
            self.item_list.append(item)

    def remove(self, item):
        try:
            index = self.item_index_map.pop(item)
        except KeyError:
            raise KeyError(item)
        self.item_list[index] = _MISSING
        self._add_dead(index)
        self._cull()

    def discard(self, item):
        try:
            index = self.item_index_map.pop(item)
        except KeyError:
            return
        self.item_list[index] = _MISSING
        self._add_dead(index)
        self._cull()

    def clear(self):
        self.item_index_map.clear()
        del self.item_list[:]
        del self.dead_indices[:]

    def isdisjoint(self, other):
        return not any(item in self.item_index_map for item in other)

    def issubset(self, other):
        if isinstance(other, _ComplementSet):
            return self.isdisjoint(other.wrapped)
        try:
            other_len = len(other)
        except TypeError:
            other = set(other)
            other_len = len(other)
        return len(self) <= other_len and all(item in other for item in self)

    def issuperset(self, other):
        if isinstance(other, _ComplementSet):
            return False
        try:
            other_len = len(other)
        except TypeError:
            other = list(other)
            other_len = len(other)
        return len(self) >= other_len and all(item in self for item in other)

    def union(self, *others):
        return self.from_iterable(chain(self, *others))

    def intersection(self, *others):
        if not others:
            return self.copy()
        other_sets = [set(other) for other in others]
        return self.from_iterable(
            item for item in self if all(item in other for other in other_sets))

    def difference(self, *others):
        if not others:
            return self.copy()
        remove = set()
        for other in others:
            remove.update(other)
        return self.from_iterable(item for item in self if item not in remove)

    def symmetric_difference(self, other):
        other_items = self.from_iterable(other)
        current = self.item_index_map
        return self.from_iterable(
            chain((item for item in self if item not in other_items),
                  (item for item in other_items if item not in current)))

    def update(self, *others):
        for other in others:
            for item in other:
                self.add(item)

    def intersection_update(self, *others):
        if not others:
            return
        other_sets = [set(other) for other in others]
        to_remove = {
            item for item in self
            if not all(item in other for other in other_sets)
        }
        if to_remove:
            self._bulk_discard(to_remove)

    def difference_update(self, *others):
        if not others:
            return
        to_remove = set()
        for other in others:
            to_remove.update(other)
        if to_remove:
            self._bulk_discard(to_remove)

    def symmetric_difference_update(self, other):
        other_items = self.from_iterable(other)
        original = set(self)
        self._bulk_discard(other_items)
        for item in other_items:
            if item not in original:
                self.add(item)

    def copy(self):
        return self.from_iterable(self)

    def __or__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__ror__(self)
        return self.union(other)

    def __ror__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__or__(self)
        return self.from_iterable(chain(other, self))

    def __and__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__rand__(self)
        return self.intersection(other)

    def __rand__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__and__(self)
        return self.from_iterable(item for item in other if item in self)

    def __sub__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__rsub__(self)
        return self.difference(other)

    def __rsub__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__sub__(self)
        return self.from_iterable(item for item in other if item not in self)

    def __xor__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__rxor__(self)
        return self.symmetric_difference(other)

    def __rxor__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__xor__(self)
        return self.from_iterable(chain(
            (item for item in other if item not in self),
            (item for item in self if item not in other)))

    def __ior__(self, other):
        self.update(other)
        return self

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return self.from_iterable(islice(self, start, stop, step))
        index = operator.index(index)
        return self.item_list[self._get_real_index(index)]

    def __delitem__(self, index):
        if isinstance(index, slice):
            for item in list(self[index]):
                self.discard(item)
            return
        self.discard(self[index])

    def pop(self, index=None):
        if not self.item_index_map:
            raise KeyError('set is empty')
        if index is None:
            index = -1
        item = self[index]
        self.discard(item)
        return item

    def count(self, item):
        return int(item in self.item_index_map)

    def index(self, item):
        try:
            real_index = self.item_index_map[item]
        except KeyError:
            raise ValueError('{!r} is not in IndexedSet'.format(item))
        return self._get_apparent_index(real_index)

    def reverse(self):
        items = list(reversed(self))
        self.item_list[:] = items
        self.item_index_map.clear()
        self.item_index_map.update(
            (item, index) for index, item in enumerate(items))
        del self.dead_indices[:]

    def sort(self, **kwargs):
        items = sorted(self, **kwargs)
        self.item_list[:] = items
        self.item_index_map.clear()
        self.item_index_map.update(
            (item, index) for index, item in enumerate(items))
        del self.dead_indices[:]


class _ComplementSet:
    __slots__ = ('wrapped',)

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __contains__(self, item):
        return item not in self.wrapped

    def __repr__(self):
        return 'complement({!r})'.format(self.wrapped)

    def __eq__(self, other):
        return isinstance(other, _ComplementSet) and self.wrapped == other.wrapped

    def __ne__(self, other):
        return not self == other

    def __or__(self, other):
        if isinstance(other, _ComplementSet):
            return complement(self.wrapped & other.wrapped)
        return complement(self.wrapped - other)

    def __ror__(self, other):
        if isinstance(other, _ComplementSet):
            return self.__or__(other)
        return complement(self.wrapped - other)

    def __and__(self, other):
        if isinstance(other, _ComplementSet):
            return complement(self.wrapped | other.wrapped)
        return other - self.wrapped

    def __rand__(self, other):
        if isinstance(other, _ComplementSet):
            return self.__and__(other)
        return other - self.wrapped

    def __sub__(self, other):
        if isinstance(other, _ComplementSet):
            return other.wrapped - self.wrapped
        return complement(self.wrapped | other)

    def __rsub__(self, other):
        if isinstance(other, _ComplementSet):
            return other.__sub__(self)
        return other & self.wrapped

    def __xor__(self, other):
        if isinstance(other, _ComplementSet):
            return self.wrapped ^ other.wrapped
        return complement(self.wrapped ^ other)

    def __rxor__(self, other):
        if isinstance(other, _ComplementSet):
            return self.__xor__(other)
        return complement(other ^ self.wrapped)

    def __le__(self, other):
        if isinstance(other, _ComplementSet):
            return other.wrapped <= self.wrapped
        return False

    def __lt__(self, other):
        if isinstance(other, _ComplementSet):
            return other.wrapped < self.wrapped
        return False

    def __ge__(self, other):
        if isinstance(other, _ComplementSet):
            return other.wrapped >= self.wrapped
        return all(item not in self.wrapped for item in other)

    def __gt__(self, other):
        if isinstance(other, _ComplementSet):
            return other.wrapped > self.wrapped
        return all(item not in self.wrapped for item in other)

    def isdisjoint(self, other):
        if isinstance(other, _ComplementSet):
            return False
        return all(item in self.wrapped for item in other)

    def issubset(self, other):
        return self <= other

    def issuperset(self, other):
        return self >= other


def complement(wrapped):
    if isinstance(wrapped, _ComplementSet):
        return wrapped.wrapped
    return _ComplementSet(wrapped)