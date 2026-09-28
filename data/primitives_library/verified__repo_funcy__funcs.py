from operator import __not__
from functools import partial, reduce, wraps

from ._inspect import get_spec, Spec
from .primitives import EMPTY
from .funcmakers import make_func, make_pred


__all__ = [
    'identity', 'constantly', 'caller',
    'reduce', 'partial',
    'rpartial', 'func_partial',
    'curry', 'rcurry', 'autocurry',
    'iffy',
    'compose', 'rcompose', 'complement', 'juxt', 'ljuxt',
]


def identity(x):
    """Returns its argument."""
    return x


def constantly(x):
    """Creates a function accepting any args, but always returning x."""
    def constant(*args, **kwargs):
        return x
    return constant


def caller(*a, **kw):
    """Creates a function calling its sole argument with given *a, **kw."""
    def call(func):
        return func(*a, **kw)
    return call


def func_partial(func, *args, **kwargs):
    """A functools.partial alternative, which returns a real function.
       Can be used to construct methods."""
    def applied(*more_args, **more_kwargs):
        merged = dict(kwargs)
        merged.update(more_kwargs)
        return func(*(args + more_args), **merged)
    return applied


def rpartial(func, *args, **kwargs):
    """Partially applies last arguments.
       New keyworded arguments extend and override kwargs."""
    def applied(*more_args, **more_kwargs):
        merged = dict(kwargs)
        merged.update(more_kwargs)
        return func(*(more_args + args), **merged)
    return applied


def curry(func, n=EMPTY):
    """Curries func into a chain of one argument functions."""
    count = get_spec(func).max_n if n is EMPTY else n
    if count <= 1:
        return func
    if count == 2:
        def first(x):
            def second(y):
                return func(x, y)
            return second
        return first

    def first(x):
        return curry(partial(func, x), count - 1)
    return first


def rcurry(func, n=EMPTY):
    """Curries func into a chain of one argument functions.
       Arguments are passed from right to left."""
    count = get_spec(func).max_n if n is EMPTY else n
    if count <= 1:
        return func
    if count == 2:
        def first(x):
            def second(y):
                return func(y, x)
            return second
        return first

    def first(x):
        return rcurry(rpartial(func, x), count - 1)
    return first


def autocurry(func, n=EMPTY, _spec=None, _args=(), _kwargs={}):
    """Creates a version of func returning its partial applications
       until sufficient arguments are passed."""
    if _spec is not None:
        spec = _spec
    elif n is EMPTY:
        spec = get_spec(func)
    else:
        spec = Spec(n, set(), n, set(), False)

    @wraps(func)
    def curried(*a, **kw):
        collected_args = _args + a
        collected_kwargs = _kwargs.copy()
        collected_kwargs.update(kw)

        total = len(collected_args) + len(collected_kwargs)
        named = len(set(collected_kwargs) & spec.names)
        required = len(set(collected_kwargs) & spec.req_names)

        if not spec.varkw and total >= spec.max_n:
            return func(*collected_args, **collected_kwargs)
        if len(collected_args) + named >= spec.max_n:
            return func(*collected_args, **collected_kwargs)
        if len(collected_args) + required >= spec.req_n:
            try:
                return func(*collected_args, **collected_kwargs)
            except TypeError:
                pass

        return autocurry(
            func,
            _spec=spec,
            _args=collected_args,
            _kwargs=collected_kwargs,
        )

    return curried


def iffy(pred, action=EMPTY, default=identity):
    """Creates a function, which conditionally applies action or default."""
    if action is EMPTY:
        return iffy(bool, pred, default)

    test = make_pred(pred)
    transform = make_func(action)

    def conditional(value):
        if test(value):
            return transform(value)
        if callable(default):
            return default(value)
        return default

    return conditional


def compose(*fs):
    """Composes passed functions."""
    if not fs:
        return identity

    functions = [make_func(f) for f in fs]

    def join(outer, inner):
        def combined(*args, **kwargs):
            return outer(inner(*args, **kwargs))
        return combined

    return reduce(join, functions)


def rcompose(*fs):
    """Composes functions, calling them from left to right."""
    return compose(*reversed(fs))


def complement(pred):
    """Constructs a complementary predicate."""
    return compose(__not__, pred)


def ljuxt(*fs):
    """Constructs a juxtaposition of the given functions.
       Result returns a list of results of fs."""
    functions = [make_func(f) for f in fs]

    def juxtaposed(*args, **kwargs):
        return [func(*args, **kwargs) for func in functions]

    return juxtaposed


def juxt(*fs):
    """Constructs a lazy juxtaposition of the given functions.
       Result returns an iterator of results of fs."""
    functions = [make_func(f) for f in fs]

    def juxtaposed(*args, **kwargs):
        return (func(*args, **kwargs) for func in functions)

    return juxtaposed