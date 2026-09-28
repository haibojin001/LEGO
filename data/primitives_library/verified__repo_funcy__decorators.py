from contextlib import ContextDecorator, contextmanager
import functools
import inspect
from inspect import unwrap

from .colls import omit

__all__ = ['decorator', 'wraps', 'unwrap', 'ContextDecorator', 'contextmanager']


def decorator(deco):
    if has_single_arg(deco):
        return make_decorator(deco)

    if has_1pos_and_kwonly(deco):
        def factory(_func=None, **kwargs):
            if _func is not None:
                return make_decorator(deco, (), kwargs)(_func)
            return make_decorator(deco, (), kwargs)
    else:
        def factory(*args, **kwargs):
            return make_decorator(deco, args, kwargs)

    return wraps(deco)(factory)


def make_decorator(deco, dargs=(), dkwargs={}):
    @wraps(deco)
    def apply(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            return deco(Call(func, args, kwargs), *dargs, **dkwargs)
        return wrapped

    apply._func = deco
    apply._args = dargs
    apply._kwargs = dkwargs
    return apply


class Call:
    def __init__(self, func, args, kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            return self._func(*self._args, **self._kwargs)
        merged_kwargs = dict(self._kwargs, **kwargs)
        return self._func(*(self._args + args), **merged_kwargs)

    def __getattr__(self, name):
        try:
            value = arggetter(self._func)(name, self._args, self._kwargs)
        except TypeError as exc:
            raise AttributeError(*exc.args)
        self.__dict__[name] = value
        return value

    def __str__(self):
        name = getattr(self._func, '__qualname__', str(self._func))
        values = [str(arg) for arg in self._args]
        values.extend('%s=%s' % item for item in self._kwargs.items())
        return '%s(%s)' % (name, ', '.join(values))

    def __repr__(self):
        return '<Call %s>' % self


def has_single_arg(func):
    parameters = tuple(inspect.signature(func).parameters.values())
    if len(parameters) != 1:
        return False
    parameter = parameters[0]
    return parameter.kind not in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )


def has_1pos_and_kwonly(func):
    parameters = inspect.signature(func).parameters.values()
    positional = 0
    varargs = 0
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs += 1
    return positional == 1 and varargs == 0


def get_argnames(func):
    original = getattr(func, '__original__', None) or unwrap(func)
    return original.__code__.co_varnames[:original.__code__.co_argcount]


def arggetter(func, _cache={}):
    if func in _cache:
        return _cache[func]

    original = getattr(func, '__original__', None) or unwrap(func)
    code = original.__code__

    positional_names = code.co_varnames[:code.co_argcount]
    offset = code.co_argcount
    keyword_only_names = code.co_varnames[
        offset:offset + code.co_kwonlyargcount
    ]
    offset += code.co_kwonlyargcount

    if hasattr(code, 'co_posonlyargcount'):
        keyword_names = positional_names[code.co_posonlyargcount:] + keyword_only_names
    else:
        keyword_names = positional_names + keyword_only_names

    varargs_name = None
    varkw_name = None
    if code.co_flags & inspect.CO_VARARGS:
        varargs_name = code.co_varnames[offset]
        offset += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        varkw_name = code.co_varnames[offset]

    names = set(code.co_varnames)
    positions = {name: index for index, name in enumerate(positional_names)}

    defaults = {}
    if original.__defaults__:
        defaults.update(
            zip(positional_names[-len(original.__defaults__):], original.__defaults__)
        )
    if original.__kwdefaults__:
        defaults.update(original.__kwdefaults__)

    def get_argument(name, args, kwargs):
        if name not in names:
            raise TypeError(
                "%s() doesn't have argument named %s" % (func.__name__, name)
            )

        index = positions.get(name)
        if index is not None and index < len(args):
            return args[index]
        if name in kwargs and name in keyword_names:
            return kwargs[name]
        if name == varargs_name:
            return args[len(positional_names):]
        if name == varkw_name:
            return omit(kwargs, keyword_names)
        if name in defaults:
            return defaults[name]

        raise TypeError(
            "%s() missing required argument: '%s'" % (func.__name__, name)
        )

    _cache[func] = get_argument
    return get_argument


def update_wrapper(wrapper, wrapped,
                   assigned=functools.WRAPPER_ASSIGNMENTS,
                   updated=functools.WRAPPER_UPDATES):
    functools.update_wrapper(wrapper, wrapped, assigned, updated)
    wrapper.__original__ = getattr(wrapped, '__original__', None) or unwrap(wrapped)
    return wrapper


update_wrapper.__doc__ = functools.update_wrapper.__doc__


def wraps(wrapped,
          assigned=functools.WRAPPER_ASSIGNMENTS,
          updated=functools.WRAPPER_UPDATES):
    return functools.partial(
        update_wrapper,
        wrapped=wrapped,
        assigned=assigned,
        updated=updated,
    )


wraps.__doc__ = functools.wraps.__doc__