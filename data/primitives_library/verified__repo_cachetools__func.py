"""Memoizing decorators compatible with functools.lru_cache."""

__all__ = ("fifo_cache", "lfu_cache", "lru_cache", "rr_cache", "ttl_cache")

import math
import random
import time
from threading import Condition

from . import FIFOCache, LFUCache, LRUCache, RRCache, TTLCache, cached, keys


class _UnboundTTLCache(TTLCache):
    def __init__(self, ttl, timer):
        super().__init__(math.inf, ttl, timer)

    @property
    def maxsize(self):
        return None


def _cache(cache, maxsize, typed):
    def decorate(function):
        make_key = keys.typedkey if typed else keys.hashkey
        wrapped = cached(
            cache=cache,
            key=make_key,
            condition=Condition(),
            info=True,
        )(function)
        wrapped.cache_parameters = lambda: {
            "maxsize": maxsize,
            "typed": typed,
        }
        return wrapped

    return decorate


def fifo_cache(maxsize=128, typed=False):
    """Memoize calls using a first-in, first-out replacement policy."""
    if maxsize is None:
        return _cache({}, None, typed)
    if callable(maxsize):
        return _cache(FIFOCache(128), 128, typed)(maxsize)
    return _cache(FIFOCache(maxsize), maxsize, typed)


def lfu_cache(maxsize=128, typed=False):
    """Memoize calls using a least-frequently-used replacement policy."""
    if maxsize is None:
        return _cache({}, None, typed)
    if callable(maxsize):
        return _cache(LFUCache(128), 128, typed)(maxsize)
    return _cache(LFUCache(maxsize), maxsize, typed)


def lru_cache(maxsize=128, typed=False):
    """Memoize calls using a least-recently-used replacement policy."""
    if maxsize is None:
        return _cache({}, None, typed)
    if callable(maxsize):
        return _cache(LRUCache(128), 128, typed)(maxsize)
    return _cache(LRUCache(maxsize), maxsize, typed)


def rr_cache(maxsize=128, choice=random.choice, typed=False):
    """Memoize calls using a random replacement policy."""
    if maxsize is None:
        return _cache({}, None, typed)
    if callable(maxsize):
        return _cache(RRCache(128, choice), 128, typed)(maxsize)
    return _cache(RRCache(maxsize, choice), maxsize, typed)


def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    """Memoize calls using an LRU policy with expiration times."""
    if maxsize is None:
        return _cache(_UnboundTTLCache(ttl, timer), None, typed)
    if callable(maxsize):
        return _cache(TTLCache(128, ttl, timer), 128, typed)(maxsize)
    return _cache(TTLCache(maxsize, ttl, timer), maxsize, typed)