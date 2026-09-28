from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, partial
from functools import update_wrapper as _stdlib_update_wrapper

__all__ = ["update_wrapper", "wraps"]


def update_wrapper(wrapper, wrapped, *args, **kwargs):
    result = _stdlib_update_wrapper(wrapper, wrapped, *args, **kwargs)
    result.__wrapped__ = wrapped
    return result


def wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):
    return partial(
        update_wrapper,
        wrapped=wrapped,
        assigned=assigned,
        updated=updated,
    )


def reraise(tp, value, tb=None):
    if value.__traceback__ is not tb:
        raise value.with_traceback(tb)
    raise value