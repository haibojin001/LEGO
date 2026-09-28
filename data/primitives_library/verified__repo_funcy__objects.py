from inspect import isclass, ismodule

from .strings import cut_prefix


__all__ = ['cached_property', 'cached_readonly', 'wrap_prop', 'monkey', 'LazyObject']


class cached_property(object):
    fset = None
    fdel = None

    def __init__(self, fget):
        self.fget = fget
        self.__doc__ = getattr(fget, '__doc__')

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        value = self.fget(instance)
        instance.__dict__[self.fget.__name__] = value
        return value


class cached_readonly(cached_property):
    def __set__(self, instance, value):
        raise AttributeError('property is read-only')


def wrap_prop(ctx):
    def decorate(prop):
        class WrappedProperty(object):
            def __repr__(self):
                return repr(prop)

            def __get__(self, instance, type=None):
                if instance is None:
                    return self
                with ctx:
                    return prop.__get__(instance, type)

            if hasattr(prop, '__set__'):
                def __set__(self, name, value):
                    with ctx:
                        return prop.__set__(name, value)

            if hasattr(prop, '__del__'):
                def __del__(self, name):
                    with ctx:
                        return prop.__del__(name)

        return WrappedProperty()
    return decorate


def monkey(cls, name=None):
    assert isclass(cls) or ismodule(cls), (
        'Attempting to monkey patch non-class and non-module'
    )

    def decorate(value):
        func = getattr(value, 'fget', value)
        func_name = name or cut_prefix(func.__name__, '%s__' % cls.__name__)

        func.__name__ = func_name
        func.original = getattr(cls, func_name, None)
        setattr(cls, func_name, value)
        return value

    return decorate


class LazyObject(object):
    def __init__(self, init):
        self.__dict__['_init'] = init

    def _setup(self):
        obj = self._init()
        object.__setattr__(self, '__class__', obj.__class__)
        object.__setattr__(self, '__dict__', obj.__dict__)

    def __getattr__(self, name):
        self._setup()
        return getattr(self, name)

    def __setattr__(self, name, value):
        self._setup()
        return setattr(self, name, value)