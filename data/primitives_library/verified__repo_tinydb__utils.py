from collections import OrderedDict, abc
from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")
D = TypeVar("D")
T = TypeVar("T")

__all__ = ("LRUCache", "freeze", "with_typehint")


def with_typehint(baseclass: type[T]):
    if TYPE_CHECKING:
        return baseclass
    return object


class LRUCache(abc.MutableMapping, Generic[K, V]):
    def __init__(self, capacity=None) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[K, V] = OrderedDict()

    @property
    def lru(self) -> list[K]:
        return list(self.cache.keys())

    @property
    def length(self) -> int:
        return len(self.cache)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return self.length

    def __contains__(self, key: object) -> bool:
        return key in self.cache

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def __delitem__(self, key: K) -> None:
        del self.cache[key]

    def __getitem__(self, key) -> V:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self) -> Iterator[K]:
        return iter(self.cache)

    def get(self, key: K, default: Optional[D] = None) -> Optional[Union[V, D]]:
        result = self.cache.get(key)
        if result is None:
            return default
        self.cache.move_to_end(key, last=True)
        return result

    def set(self, key: K, value: V):
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key, last=True)
            return

        self.cache[key] = value
        if self.capacity is not None and self.length > self.capacity:
            self.cache.popitem(last=False)


class FrozenDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def _immutable(self, *args, **kwargs):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    setdefault = _immutable
    popitem = _immutable

    def update(self, e=None, **f):
        raise TypeError("object is immutable")

    def pop(self, k, d=None):
        raise TypeError("object is immutable")


def freeze(obj):
    if isinstance(obj, dict):
        return FrozenDict((key, freeze(value)) for key, value in obj.items())
    if isinstance(obj, list):
        return tuple(freeze(value) for value in obj)
    if isinstance(obj, tuple):
        return tuple(freeze(value) for value in obj)
    if isinstance(obj, set):
        return frozenset(obj)
    return obj