import functools

__all__ = ()


def _condition_info(func, cache, key, lock, cond, info):
    hits = 0
    misses = 0
    pending = set()

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = key(*args, **kwargs)
        with lock:
            cond.wait_for(lambda: cache_key not in pending)
            try:
                value = cache[cache_key]
            except KeyError:
                pending.add(cache_key)
                misses += 1
            else:
                hits += 1
                return value
        try:
            value = func(*args, **kwargs)
            with lock:
                try:
                    cache[cache_key] = value
                except ValueError:
                    pass
                return value
        finally:
            with lock:
                pending.remove(cache_key)
                cond.notify_all()

    def cache_clear():
        nonlocal hits, misses
        with lock:
            cache.clear()
            hits = 0
            misses = 0

    def cache_info():
        with lock:
            return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _locked_info(func, cache, key, lock, info):
    hits = 0
    misses = 0

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = key(*args, **kwargs)
        with lock:
            try:
                value = cache[cache_key]
            except KeyError:
                misses += 1
            else:
                hits += 1
                return value
        value = func(*args, **kwargs)
        with lock:
            try:
                return cache.setdefault(cache_key, value)
            except ValueError:
                return value

    def cache_clear():
        nonlocal hits, misses
        with lock:
            cache.clear()
            hits = 0
            misses = 0

    def cache_info():
        with lock:
            return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _unlocked_info(func, cache, key, info):
    hits = 0
    misses = 0

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = key(*args, **kwargs)
        try:
            value = cache[cache_key]
        except KeyError:
            misses += 1
        else:
            hits += 1
            return value
        value = func(*args, **kwargs)
        try:
            cache[cache_key] = value
        except ValueError:
            pass
        return value

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_info():
        return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _uncached_info(func, info):
    misses = 0

    def wrapper(*args, **kwargs):
        nonlocal misses
        misses += 1
        return func(*args, **kwargs)

    def cache_clear():
        nonlocal misses
        misses = 0

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = lambda: info(0, misses)
    return wrapper


def _condition(func, cache, key, lock, cond):
    pending = set()

    def wrapper(*args, **kwargs):
        cache_key = key(*args, **kwargs)
        with lock:
            cond.wait_for(lambda: cache_key not in pending)
            try:
                return cache[cache_key]
            except KeyError:
                pending.add(cache_key)
        try:
            value = func(*args, **kwargs)
            with lock:
                try:
                    cache[cache_key] = value
                except ValueError:
                    pass
                return value
        finally:
            with lock:
                pending.remove(cache_key)
                cond.notify_all()

    def cache_clear():
        with lock:
            cache.clear()

    wrapper.cache_clear = cache_clear
    return wrapper


def _locked(func, cache, key, lock):
    def wrapper(*args, **kwargs):
        cache_key = key(*args, **kwargs)
        with lock:
            try:
                return cache[cache_key]
            except KeyError:
                pass
        value = func(*args, **kwargs)
        with lock:
            try:
                return cache.setdefault(cache_key, value)
            except ValueError:
                return value

    def cache_clear():
        with lock:
            cache.clear()

    wrapper.cache_clear = cache_clear
    return wrapper


def _unlocked(func, cache, key):
    def wrapper(*args, **kwargs):
        cache_key = key(*args, **kwargs)
        try:
            return cache[cache_key]
        except KeyError:
            pass
        value = func(*args, **kwargs)
        try:
            cache[cache_key] = value
        except ValueError:
            pass
        return value

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def _uncached(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.cache_clear = lambda: None
    return wrapper


def _wrapper(func, cache, key, lock=None, cond=None, info=None):
    if info is None:
        if cache is None:
            wrapper = _uncached(func)
        elif cond is not None:
            if lock is None:
                wrapper = _condition(func, cache, key, cond, cond)
            else:
                wrapper = _condition(func, cache, key, lock, cond)
        elif lock is not None:
            wrapper = _locked(func, cache, key, lock)
        else:
            wrapper = _unlocked(func, cache, key)
        wrapper.cache_info = None
    else:
        if cache is None:
            wrapper = _uncached_info(func, info)
        elif cond is not None:
            if lock is None:
                wrapper = _condition_info(func, cache, key, cond, cond, info)
            else:
                wrapper = _condition_info(func, cache, key, lock, cond, info)
        elif lock is not None:
            wrapper = _locked_info(func, cache, key, lock, info)
        else:
            wrapper = _unlocked_info(func, cache, key, info)

    wrapper.cache = cache
    wrapper.cache_key = key
    wrapper.cache_lock = lock if lock is not None else cond
    wrapper.cache_condition = cond

    return functools.update_wrapper(wrapper, func)