import functools
import warnings
import weakref
from collections import namedtuple

try:
    from . import keys
except ImportError:
    keys = None

try:
    from ._cached import _CacheInfo
except ImportError:
    _CacheInfo = namedtuple("CacheInfo", ("hits", "misses", "maxsize", "currsize"))

__all__ = ()


def _warn_classmethod(stacklevel):
    warnings.warn(
        "decorating class methods with @cachedmethod is deprecated",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _warn_instance_dict(msg, stacklevel):
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def _none(_):
    return None


def _cache_info(cache, hits, misses):
    return _CacheInfo(hits, misses, cache.maxsize, cache.currsize)


class _WrapperBase:
    def __init__(self, obj, method, cache, key, lock=None, cond=None):
        if isinstance(obj, type):
            _warn_classmethod(stacklevel=5)
        functools.update_wrapper(self, method)
        self._obj = obj
        self.__cache = cache
        self.__key = functools.partial(key, obj)
        self.__lock = lock if lock is not None else _none
        self.__cond = cond if cond is not None else _none

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def cache_clear(self):
        raise NotImplementedError()

    @property
    def cache(self):
        return self.__cache(self._obj)

    @property
    def cache_key(self):
        return self.__key

    @property
    def cache_lock(self):
        return self.__lock(self._obj)

    @property
    def cache_condition(self):
        return self.__cond(self._obj)


class _DescriptorBase:
    def __init__(self, deprecated=False):
        self.__attrname = None
        self.__deprecated = deprecated

    def __set_name__(self, owner, name):
        if self.__attrname is None:
            self.__attrname = name
        elif name != self.__attrname:
            raise TypeError(
                "Cannot assign the same @cachedmethod to two different names "
                f"({self.__attrname!r} and {name!r})."
            )

    def __get__(self, obj, objtype=None):
        wrapper = self.Wrapper(obj)
        if obj is None:
            pass
        elif self.__attrname is not None:
            try:
                wrapper = obj.__dict__.setdefault(self.__attrname, wrapper)
            except AttributeError:
                msg = (
                    f"No '__dict__' attribute on {type(obj).__name__!r} "
                    f"instance to cache {self.__attrname!r} property."
                )
                if self.__deprecated:
                    _warn_instance_dict(msg, 3)
                else:
                    raise TypeError(msg) from None
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(obj).__name__!r} "
                    f"instance does not support item assignment for caching "
                    f"{self.__attrname!r} property."
                )
                if self.__deprecated:
                    _warn_instance_dict(msg, 3)
                else:
                    raise TypeError(msg) from None
        elif self.__deprecated:
            pass
        else:
            raise TypeError(
                "Cannot use @cachedmethod instance without calling "
                "__set_name__ on it"
            ) from None
        return wrapper


class _DeprecatedDescriptorBase(_DescriptorBase):
    def __init__(self, wrapper, cache_clear):
        super().__init__(deprecated=True)
        self.__wrapper = wrapper
        self.__cache_clear = cache_clear

    def __call__(self, *args, **kwargs):
        _warn_classmethod(stacklevel=3)
        return self.__wrapper(*args, **kwargs)

    def cache_clear(self, objtype):
        _warn_classmethod(stacklevel=3)
        return self.__cache_clear(objtype)


def _condition_info(method, cache, key, lock, cond, info):
    class Descriptor(_DescriptorBase):
        class Wrapper(_WrapperBase):
            def __init__(self, obj):
                super().__init__(obj, method, cache, key, lock, cond)
                self.__hits = 0
                self.__misses = 0
                self.__pending = set()

            def __call__(self, *args, **kwargs):
                cache_ = self.cache
                lock_ = self.cache_lock
                cond_ = self.cache_condition
                key_ = self.cache_key(*args, **kwargs)
                with lock_:
                    cond_.wait_for(lambda: key_ not in self.__pending)
                    try:
                        value = cache_[key_]
                        self.__hits += 1
                        return value
                    except KeyError:
                        self.__pending.add(key_)
                        self.__misses += 1
                try:
                    value = method(self._obj, *args, **kwargs)
                    with lock_:
                        try:
                            cache_[key_] = value
                        except ValueError:
                            pass
                        return value
                finally:
                    with lock_:
                        self.__pending.remove(key_)
                        cond_.notify_all()

            def cache_clear(self):
                with self.cache_lock:
                    self.cache.clear()
                    self.__hits = 0
                    self.__misses = 0

            def cache_info(self):
                with self.cache_lock:
                    return info(self.cache, self.__hits, self.__misses)

    return Descriptor()


def _locked_info(method, cache, key, lock, info):
    class Descriptor(_DescriptorBase):
        class Wrapper(_WrapperBase):
            def __init__(self, obj):
                super().__init__(obj, method, cache, key, lock)
                self.__hits = 0
                self.__misses = 0

            def __call__(self, *args, **kwargs):
                cache_ = self.cache
                lock_ = self.cache_lock
                key_ = self.cache_key(*args, **kwargs)
                with lock_:
                    try:
                        value = cache_[key_]
                        self.__hits += 1
                        return value
                    except KeyError:
                        self.__misses += 1
                value = method(self._obj, *args, **kwargs)
                with lock_:
                    try:
                        return cache_.setdefault(key_, value)
                    except ValueError:
                        return value

            def cache_clear(self):
                with self.cache_lock:
                    self.cache.clear()
                    self.__hits = 0
                    self.__misses = 0

            def cache_info(self):
                with self.cache_lock:
                    return info(self.cache, self.__hits, self.__misses)

    return Descriptor()


def _unlocked_info(method, cache, key, info):
    class Descriptor(_DescriptorBase):
        class Wrapper(_WrapperBase):
            def __init__(self, obj):
                super().__init__(obj, method, cache, key)
                self.__hits = 0
                self.__misses = 0

            def __call__(self, *args, **kwargs):
                cache_ = self.cache
                key_ = self.cache_key(*args, **kwargs)
                try:
                    value = cache_[key_]
                    self.__hits += 1
                    return value
                except KeyError:
                    self.__misses += 1
                value = method(self._obj, *args, **kwargs)
                try:
                    cache_[key_] = value
                except ValueError:
                    pass
                return value

            def cache_clear(self):
                self.cache.clear()
                self.__hits = 0
                self.__misses = 0

            def cache_info(self):
                return info(self.cache, self.__hits, self.__misses)

    return Descriptor()


def _condition(method, cache, key, lock, cond):
    class Descriptor(_DescriptorBase):
        class Wrapper(_WrapperBase):
            def __init__(self, obj):
                super().__init__(obj, method, cache, key, lock, cond)
                self.__pending = set()

            def __call__(self, *args, **kwargs):
                cache_ = self.cache
                lock_ = self.cache_lock
                cond_ = self.cache_condition
                key_ = self.cache_key(*args, **kwargs)
                with lock_:
                    cond_.wait_for(lambda: key_ not in self.__pending)
                    try:
                        return cache_[key_]
                    except KeyError:
                        self.__pending.add(key_)
                try:
                    value = method(self._obj, *args, **kwargs)
                    with lock_:
                        try:
                            cache_[key_] = value
                        except ValueError:
                            pass
                        return value
                finally:
                    with lock_:
                        self.__pending.remove(key_)
                        cond_.notify_all()

            def cache_clear(self):
                with self.cache_lock:
                    self.cache.clear()

    return Descriptor()


def _locked(method, cache, key, lock):
    class Descriptor(_DescriptorBase):
        class Wrapper(_WrapperBase):
            def __init__(self, obj):
                super().__init__(obj, method, cache, key, lock)

            def __call__(self, *args, **kwargs):
                cache_ = self.cache
                lock_ = self.cache_lock
                key_ = self.cache_key(*args, **kwargs)
                with lock_:
                    try:
                        return cache_[key_]
                    except KeyError:
                        pass
                value = method(self._obj, *args, **kwargs)
                with lock_:
                    try:
                        return cache_.setdefault(key_, value)
                    except ValueError:
                        return value

            def cache_clear(self):
                with self.cache_lock:
                    self.cache.clear()

    return Descriptor()


def _unlocked(method, cache, key):
    class Descriptor(_DescriptorBase):
        class Wrapper(_WrapperBase):
            def __init__(self, obj):
                super().__init__(obj, method, cache, key)

            def __call__(self, *args, **kwargs):
                cache_ = self.cache
                key_ = self.cache_key(*args, **kwargs)
                try:
                    return cache_[key_]
                except KeyError:
                    pass
                value = method(self._obj, *args, **kwargs)
                try:
                    cache_[key_] = value
                except ValueError:
                    pass
                return value

            def cache_clear(self):
                self.cache.clear()

    return Descriptor()


def cachedmethod(cache, key=None, lock=None, condition=None, info=False):
    if key is None:
        if keys is None:
            raise ImportError("cannot import name 'keys'")
        key = keys.methodkey

    def decorator(method):
        if info is True:
            info_ = _cache_info
        else:
            info_ = info

        if info_:
            if condition is not None:
                return _condition_info(method, cache, key, lock, condition, info_)
            if lock is not None:
                return _locked_info(method, cache, key, lock, info_)
            return _unlocked_info(method, cache, key, info_)

        if condition is not None:
            return _condition(method, cache, key, lock, condition)
        if lock is not None:
            return _locked(method, cache, key, lock)
        return _unlocked(method, cache, key)

    return decorator