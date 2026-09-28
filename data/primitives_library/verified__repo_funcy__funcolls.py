from .funcs import compose, juxt
from .colls import some, none, one


__all__ = ['all_fn', 'any_fn', 'none_fn', 'one_fn', 'some_fn']


def _predicate_group(reducer, functions):
    return compose(reducer, juxt(*functions))


def all_fn(*fs):
    """Create a predicate satisfied when every supplied predicate is true."""
    return _predicate_group(all, fs)


def any_fn(*fs):
    """Create a predicate satisfied when at least one predicate is true."""
    return _predicate_group(any, fs)


def none_fn(*fs):
    """Create a predicate satisfied when no supplied predicate is true."""
    return _predicate_group(none, fs)


def one_fn(*fs):
    """Create a predicate satisfied when precisely one predicate is true."""
    return _predicate_group(one, fs)


def some_fn(*fs):
    """Create a function returning the first truthy result from its functions."""
    return _predicate_group(some, fs)