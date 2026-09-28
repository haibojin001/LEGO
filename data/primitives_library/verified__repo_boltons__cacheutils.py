import functools
import heapq
import itertools
import weakref
from operator import attrgetter

try:
    from threading import RLock
except Exception:
    class RLock:
        def __enter__(self):
            return self

        def __exit__(self, exctype, excinst, exctb):
            return None

try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
    _KWARG_MARK = make_sentinel(var_name='_KWARG_MARK')
except ImportError:
    _MISSING = object()
    _KWARG_MARK = object()


PREV, NEXT, KEY, VALUE = range(4)
DEFAULT_MAX_SIZE = 128


class LRI(dict):
    def __init__(self, max_size=DEFAULT_MAX_SIZE, values=None,
                 on_miss=None):
        if max_size <= 0:
            raise ValueError('expected max_size > 0, not %r' % max_size)
        if on_miss is not None and not callable(on_miss):
            raise TypeError('expected on_miss to be a callable'
                            ' (or None), not %r' % on_miss)

        self.hit_count = 0
        self.miss_count = 0
        self.soft_miss_count = 0
        self.max_size = max_size
        self.on_miss = on_miss
        self._lock = RLock()
        self._init_ll()

        if values:
            self.update(values)

    def _init_ll(self):
        anchor = []
        anchor[:] = [anchor, anchor, _MISSING, _MISSING]
        self._anchor = anchor
        self._link_lookup = {}

    def _print_ll(self):
        print('***')
        for key, value in self._get_flattened_ll():
            print(key, value)
        print('***')

    def _get_flattened_ll(self):
        ret = []
        link = self._anchor
        while True:
            ret.append((link[KEY], link[VALUE]))
            link = link[NEXT]
            if link is self._anchor:
                return ret

    def _get_link_and_move_to_front_of_ll(self, key):
        link = self._link_lookup[key]

        previous = link[PREV]
        following = link[NEXT]
        previous[NEXT] = following
        following[PREV] = previous

        anchor = self._anchor
        newest = anchor[PREV]
        newest[NEXT] = link
        anchor[PREV] = link
        link[PREV] = newest
        link[NEXT] = anchor
        return link

    def _set_key_and_add_to_front_of_ll(self, key, value):
        anchor = self._anchor
        newest = anchor[PREV]
        link = [newest, anchor, key, value]
        newest[NEXT] = link
        anchor[PREV] = link
        self._link_lookup[key] = link

    def _set_key_and_evict_last_in_ll(self, key, value):
        old_anchor = self._anchor
        old_anchor[KEY] = key
        old_anchor[VALUE] = value

        anchor = old_anchor[NEXT]
        self._anchor = anchor
        old_key = anchor[KEY]
        anchor[KEY] = _MISSING
        anchor[VALUE] = _MISSING

        del self._link_lookup[old_key]
        self._link_lookup[key] = old_anchor
        return old_key

    def _remove_from_ll(self, key):
        link = self._link_lookup.pop(key)
        link[PREV][NEXT] = link[NEXT]
        link[NEXT][PREV] = link[PREV]

    def __setitem__(self, key, value):
        with self._lock:
            try:
                link = self._get_link_and_move_to_front_of_ll(key)
            except KeyError:
                if len(self) < self.max_size:
                    self._set_key_and_add_to_front_of_ll(key, value)
                else:
                    old_key = self._set_key_and_evict_last_in_ll(key, value)
                    dict.__delitem__(self, old_key)
            else:
                link[VALUE] = value
            dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        with self._lock:
            try:
                link = self._link_lookup[key]
            except KeyError:
                self.miss_count += 1
                if self.on_miss is None:
                    raise
                value = self.on_miss(key)
                self[key] = value
                return value
            self.hit_count += 1
            return link[VALUE]

    def get(self, key, default=None):
        with self._lock:
            try:
                link = self._link_lookup[key]
            except KeyError:
                self.miss_count += 1
                self.soft_miss_count += 1
                return default
            self.hit_count += 1
            return link[VALUE]

    def setdefault(self, key, default=None):
        with self._lock:
            try:
                link = self._link_lookup[key]
            except KeyError:
                self.miss_count += 1
                self.soft_miss_count += 1
                self[key] = default
                return default
            self.hit_count += 1
            return link[VALUE]

    def __delitem__(self, key):
        with self._lock:
            self._remove_from_ll(key)
            dict.__delitem__(self, key)

    def pop(self, key, default=_MISSING):
        with self._lock:
            try:
                value = dict.__getitem__(self, key)
            except KeyError:
                if default is _MISSING:
                    raise
                return default
            self._remove_from_ll(key)
            dict.__delitem__(self, key)
            return value

    def popitem(self):
        with self._lock:
            if not self:
                raise KeyError('popitem(): cache is empty')
            link = self._anchor[NEXT]
            key = link[KEY]
            value = link[VALUE]
            self._remove_from_ll(key)
            dict.__delitem__(self, key)
            return key, value

    def clear(self):
        with self._lock:
            dict.clear(self)
            self._init_ll()

    def update(self, other=(), **kwargs):
        if hasattr(other, 'keys'):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self):
        new_cache = self.__class__(self.max_size, on_miss=self.on_miss)
        for key, value in self.items():
            new_cache[key] = value
        new_cache.hit_count = self.hit_count
        new_cache.miss_count = self.miss_count
        new_cache.soft_miss_count = self.soft_miss_count
        return new_cache

    __copy__ = copy

    def __reduce__(self):
        return (self.__class__, (self.max_size, dict(self), self.on_miss))


class LRU(LRI):
    def __getitem__(self, key):
        with self._lock:
            try:
                link = self._get_link_and_move_to_front_of_ll(key)
            except KeyError:
                self.miss_count += 1
                if self.on_miss is None:
                    raise
                value = self.on_miss(key)
                self[key] = value
                return value
            self.hit_count += 1
            return link[VALUE]

    def get(self, key, default=None):
        with self._lock:
            try:
                link = self._get_link_and_move_to_front_of_ll(key)
            except KeyError:
                self.miss_count += 1
                self.soft_miss_count += 1
                return default
            self.hit_count += 1
            return link[VALUE]

    def setdefault(self, key, default=None):
        with self._lock:
            try:
                link = self._get_link_and_move_to_front_of_ll(key)
            except KeyError:
                self.miss_count += 1
                self.soft_miss_count += 1
                self[key] = default
                return default
            self.hit_count += 1
            return link[VALUE]


def make_cache_key(args, kwargs, typed=False,
                   kwarg_mark=(_KWARG_MARK,)):
    key = args
    if kwargs:
        key += kwarg_mark
        for item in kwargs.items():
            key += item
    if typed:
        key += tuple(type(value) for value in args)
        if kwargs:
            key += tuple(type(value) for _, value in kwargs.items())
    return key


def cached(cache, scoped=True, typed=False, key=None):
    def decorator(func):
        if key is None:
            def key_func(args, kwargs):
                return make_cache_key(args, kwargs, typed)
        else:
            def key_func(args, kwargs):
                return key(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key_func(args, kwargs)
            if scoped:
                if isinstance(cache_key, tuple):
                    cache_key = (func,) + cache_key
                else:
                    cache_key = (func, cache_key)
            try:
                return cache[cache_key]
            except KeyError:
                value = func(*args, **kwargs)
                cache[cache_key] = value
                return value

        wrapper.cache = cache
        wrapper.cache_key = key_func
        return wrapper
    return decorator


def cachedmethod(cache, scoped=True, typed=False, key=None):
    if isinstance(cache, str):
        cache = attrgetter(cache)

    def decorator(func):
        if key is None:
            def key_func(args, kwargs):
                return make_cache_key(args, kwargs, typed)
        else:
            def key_func(args, kwargs):
                return key(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_obj = cache(self)
            cache_key = key_func(args, kwargs)
            if scoped:
                if isinstance(cache_key, tuple):
                    cache_key = (func,) + cache_key
                else:
                    cache_key = (func, cache_key)
            try:
                return cache_obj[cache_key]
            except KeyError:
                value = func(self, *args, **kwargs)
                cache_obj[cache_key] = value
                return value

        wrapper.cache = cache
        wrapper.cache_key = key_func
        return wrapper
    return decorator


class CachedInstancePartial(functools.partial):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        try:
            cache = self._instance_cache
        except AttributeError:
            cache = self._instance_cache = weakref.WeakKeyDictionary()

        try:
            return cache[obj]
        except KeyError:
            bound = self.__class__(self.func, obj, *self.args,
                                   **(self.keywords or {}))
            cache[obj] = bound
            return bound


class ThresholdCounter:
    def __init__(self, threshold=0.001):
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError('expected threshold between 0 and 1, not %r'
                             % threshold)
        self.threshold = threshold
        self.total_count = 0
        self._count_map = {}
        self._bucket_width = max(1, int(1.0 / threshold))
        self._cur_bucket = 1

    def add(self, key, count=1):
        if count < 0:
            raise ValueError('expected count >= 0, not %r' % count)
        if count == 0:
            return

        for _ in range(count):
            self.total_count += 1
            bucket = ((self.total_count - 1) // self._bucket_width) + 1
            if key in self._count_map:
                self._count_map[key][0] += 1
            else:
                self._count_map[key] = [1, bucket - 1]

            if bucket != self._cur_bucket:
                self._cur_bucket = bucket
                self._count_map = {
                    item_key: item_value
                    for item_key, item_value in self._count_map.items()
                    if item_value[0] + item_value[1] > bucket
                }

    def update(self, iterable=None, **kwargs):
        if iterable is not None:
            if hasattr(iterable, 'items'):
                for key, count in iterable.items():
                    self.add(key, count)
            else:
                for key in iterable:
                    self.add(key)
        for key, count in kwargs.items():
            self.add(key, count)

    def get(self, key, default=0):
        try:
            return self._count_map[key][0]
        except KeyError:
            return default

    def __getitem__(self, key):
        return self._count_map[key][0]

    def __contains__(self, key):
        return key in self._count_map

    def __len__(self):
        return len(self._count_map)

    def __iter__(self):
        return iter(self._count_map)

    def items(self):
        return ((key, value[0]) for key, value in self._count_map.items())

    def keys(self):
        return self._count_map.keys()

    def values(self):
        return (value[0] for value in self._count_map.values())

    def elements(self):
        return itertools.chain.from_iterable(
            itertools.repeat(key, value[0])
            for key, value in self._count_map.items()
        )

    def most_common(self, n=None):
        items = self.items()
        if n is None:
            return sorted(items, key=lambda item: item[1], reverse=True)
        return heapq.nlargest(n, items, key=lambda item: item[1])

    def clear(self):
        self.total_count = 0
        self._count_map.clear()
        self._cur_bucket = 1


class MinIDMap:
    def __init__(self):
        self.mapping = weakref.WeakKeyDictionary()
        self.ref_map = {}
        self.next_id = 0
        self._freed = []

    def _on_collect(self, ident):
        self.ref_map.pop(ident, None)
        heapq.heappush(self._freed, ident)

    def get(self, obj):
        try:
            return self.mapping[obj]
        except KeyError:
            if self._freed:
                ident = heapq.heappop(self._freed)
            else:
                ident = self.next_id
                self.next_id += 1

            self_ref = weakref.ref(self)

            def callback(ref, ident=ident, self_ref=self_ref):
                owner = self_ref()
                if owner is not None:
                    owner._on_collect(ident)

            obj_ref = weakref.ref(obj, callback)
            self.mapping[obj] = ident
            self.ref_map[ident] = obj_ref
            return ident

    def drop(self, obj):
        ident = self.mapping.pop(obj)
        self.ref_map.pop(ident, None)
        heapq.heappush(self._freed, ident)
        return ident

    def __getitem__(self, obj):
        return self.get(obj)

    def __contains__(self, obj):
        return obj in self.mapping

    def __len__(self):
        return len(self.mapping)