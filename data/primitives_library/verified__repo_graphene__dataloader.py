from asyncio import gather, ensure_future, get_event_loop, iscoroutine, iscoroutinefunction
from collections import namedtuple
from collections.abc import Iterable
from functools import partial
from typing import List


Loader = namedtuple("Loader", "key,future")


def iscoroutinefunctionorpartial(fn):
    target = fn.func if isinstance(fn, partial) else fn
    return iscoroutinefunction(target)


class DataLoader(object):
    batch = True
    max_batch_size = None  # type: int
    cache = True

    def __init__(
        self,
        batch_load_fn=None,
        batch=None,
        max_batch_size=None,
        cache=None,
        get_cache_key=None,
        cache_map=None,
        loop=None,
    ):
        self._loop = loop

        if batch_load_fn is not None:
            self.batch_load_fn = batch_load_fn

        assert iscoroutinefunctionorpartial(
            self.batch_load_fn
        ), "batch_load_fn must be coroutine. Received: {}".format(
            self.batch_load_fn
        )

        if not callable(self.batch_load_fn):
            raise TypeError(
                (
                    "DataLoader must be have a batch_load_fn which accepts "
                    "Iterable<key> and returns Future<Iterable<value>>, but got: {}."
                ).format(batch_load_fn)
            )

        if batch is not None:
            self.batch = batch

        if max_batch_size is not None:
            self.max_batch_size = max_batch_size

        if cache is not None:
            self.cache = cache

        self.get_cache_key = get_cache_key or (lambda key: key)
        self._cache = cache_map if cache_map is not None else {}
        self._queue: List[Loader] = []

    @property
    def loop(self):
        if not self._loop:
            self._loop = get_event_loop()
        return self._loop

    def load(self, key=None):
        if key is None:
            raise TypeError(
                (
                    "The loader.load() function must be called with a value, "
                    "but got: {}."
                ).format(key)
            )

        cache_key = self.get_cache_key(key)

        if self.cache:
            result = self._cache.get(cache_key)
            if result:
                return result

        future = self.loop.create_future()

        if self.cache:
            self._cache[cache_key] = future

        self.do_resolve_reject(key, future)
        return future

    def do_resolve_reject(self, key, future):
        self._queue.append(Loader(key=key, future=future))

        if len(self._queue) == 1:
            if self.batch:
                enqueue_post_future_job(self.loop, self)
            else:
                dispatch_queue(self)

    def load_many(self, keys):
        if not isinstance(keys, Iterable):
            raise TypeError(
                (
                    "The loader.load_many() function must be called with Iterable<key> "
                    "but got: {}."
                ).format(keys)
            )

        return gather(*[self.load(key) for key in keys])

    def clear(self, key):
        self._cache.pop(self.get_cache_key(key), None)
        return self

    def clear_all(self):
        self._cache.clear()
        return self

    def prime(self, key, value):
        cache_key = self.get_cache_key(key)

        if cache_key not in self._cache:
            future = self.loop.create_future()
            if isinstance(value, Exception):
                future.set_exception(value)
            else:
                future.set_result(value)
            self._cache[cache_key] = future

        return self


def enqueue_post_future_job(loop, loader):
    async def dispatch():
        dispatch_queue(loader)

    loop.call_soon(ensure_future, dispatch())


def get_chunks(iterable_obj, chunk_size=1):
    size = max(1, chunk_size)
    return (
        iterable_obj[index : index + size]
        for index in range(0, len(iterable_obj), size)
    )


def dispatch_queue(loader):
    queue = loader._queue
    loader._queue = []

    batch_size = loader.max_batch_size
    if batch_size and batch_size < len(queue):
        for queue_chunk in get_chunks(queue, batch_size):
            ensure_future(dispatch_queue_batch(loader, queue_chunk))
    else:
        ensure_future(dispatch_queue_batch(loader, queue))


async def dispatch_queue_batch(loader, queue):
    keys = [item.key for item in queue]
    batch_future = loader.batch_load_fn(keys)

    if not batch_future or not iscoroutine(batch_future):
        return failed_dispatch(
            loader,
            queue,
            TypeError(
                (
                    "DataLoader must be constructed with a function which accepts "
                    "Iterable<key> and returns Future<Iterable<value>>, but the function did "
                    "not return a Coroutine: {}."
                ).format(batch_future)
            ),
        )

    try:
        values = await batch_future

        if not isinstance(values, Iterable):
            raise TypeError(
                (
                    "DataLoader must be constructed with a function which accepts "
                    "Iterable<key> and returns Future<Iterable<value>>, but the function did "
                    "not return a Future of a Iterable: {}."
                ).format(values)
            )

        values = list(values)

        if len(values) != len(keys):
            raise TypeError(
                (
                    "DataLoader must be constructed with a function which accepts "
                    "Iterable<key> and returns Future<Iterable<value>>, but the function did "
                    "not return a Future of a Iterable with the same length as the Iterable "
                    "of keys."
                    "\n\nKeys:\n{}"
                    "\n\nValues:\n{}"
                ).format(keys, values)
            )

        for item, value in zip(queue, values):
            if isinstance(value, Exception):
                item.future.set_exception(value)
            else:
                item.future.set_result(value)

    except Exception as error:
        return failed_dispatch(loader, queue, error)


def failed_dispatch(loader, queue, error):
    for item in queue:
        loader.clear(item.key)
        item.future.set_exception(error)