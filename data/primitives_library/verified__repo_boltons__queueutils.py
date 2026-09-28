from heapq import heappop, heappush
from bisect import insort
import itertools

try:
    from .typeutils import make_sentinel
except ImportError:
    _REMOVED = object()
else:
    _REMOVED = make_sentinel(var_name='_REMOVED')

try:
    from .listutils import BList
except ImportError:
    BList = list


__all__ = [
    'PriorityQueue',
    'BasePriorityQueue',
    'HeapPriorityQueue',
    'SortedPriorityQueue',
]


class BasePriorityQueue:
    _backend_type = list

    @staticmethod
    def _default_priority_key(priority):
        return -float(priority or 0)

    def __init__(self, **kw):
        self._pq = self._backend_type()
        self._entry_map = {}
        self._counter = itertools.count()
        self._get_priority = kw.pop('priority_key', self._default_priority_key)
        if kw:
            raise TypeError('unexpected keyword arguments: %r' % kw.keys())

    @staticmethod
    def _push_entry(backend, entry):
        pass

    @staticmethod
    def _pop_entry(backend):
        pass

    def add(self, task, priority=None):
        effective_priority = self._get_priority(priority)
        if task in self._entry_map:
            self.remove(task)
        entry = [effective_priority, next(self._counter), task]
        self._entry_map[task] = entry
        self._push_entry(self._pq, entry)

    def remove(self, task):
        entry = self._entry_map.pop(task)
        entry[-1] = _REMOVED

    def _cull(self, raise_exc=True):
        while self._pq:
            _, _, task = self._pq[0]
            if task is not _REMOVED:
                return
            self._pop_entry(self._pq)
        if raise_exc:
            raise IndexError('empty priority queue')

    def peek(self, default=_REMOVED):
        try:
            self._cull()
            _, _, task = self._pq[0]
        except IndexError:
            if default is not _REMOVED:
                return default
            raise IndexError('peek on empty queue')
        return task

    def pop(self, default=_REMOVED):
        try:
            self._cull()
            _, _, task = self._pop_entry(self._pq)
            del self._entry_map[task]
        except IndexError:
            if default is not _REMOVED:
                return default
            raise IndexError('pop on empty queue')
        return task

    def __len__(self):
        return len(self._entry_map)


class HeapPriorityQueue(BasePriorityQueue):
    @staticmethod
    def _push_entry(backend, entry):
        heappush(backend, entry)

    @staticmethod
    def _pop_entry(backend):
        return heappop(backend)


class SortedPriorityQueue(BasePriorityQueue):
    _backend_type = BList

    @staticmethod
    def _push_entry(backend, entry):
        insort(backend, entry)

    @staticmethod
    def _pop_entry(backend):
        return backend.pop(0)


PriorityQueue = SortedPriorityQueue