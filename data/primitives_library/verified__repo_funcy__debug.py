import re
import traceback
from functools import partial
from itertools import chain
from timeit import default_timer as timer

from .decorators import Call, decorator, wraps


__all__ = [
    'tap',
    'log_calls', 'print_calls',
    'log_enters', 'print_enters',
    'log_exits', 'print_exits',
    'log_errors', 'print_errors',
    'log_durations', 'print_durations',
    'log_iter_durations', 'print_iter_durations',
]


REPR_LEN = 25


def tap(x, label=None):
    """Prints x and then returns it."""
    if label:
        print('%s: %s' % (label, x))
    else:
        print(x)
    return x


@decorator
def log_calls(call, print_func, errors=True, stack=True, repr_len=REPR_LEN):
    """Logs or prints all function calls,
       including arguments, results and raised exceptions."""
    call_text = signature_repr(call, repr_len)
    try:
        print_func('Call %s' % call_text)
        value = call()
        print_func('-> %s from %s' % (smart_repr(value, max_len=None), call_text))
        return value
    except BaseException as error:
        if errors:
            print_func('-> ' + _format_error(call_text, error, stack))
        raise


def print_calls(errors=True, stack=True, repr_len=REPR_LEN):
    if callable(errors):
        return log_calls(print)(errors)
    return log_calls(print, errors, stack, repr_len)


print_calls.__doc__ = log_calls.__doc__


@decorator
def log_enters(call, print_func, repr_len=REPR_LEN):
    """Logs each entrance to a function."""
    print_func('Call %s' % signature_repr(call, repr_len))
    return call()


def print_enters(repr_len=REPR_LEN):
    """Prints on each entrance to a function."""
    if callable(repr_len):
        return log_enters(print)(repr_len)
    return log_enters(print, repr_len)


@decorator
def log_exits(call, print_func, errors=True, stack=True, repr_len=REPR_LEN):
    """Logs exits from a function."""
    call_text = signature_repr(call, repr_len)
    try:
        value = call()
        print_func('-> %s from %s' % (smart_repr(value, max_len=None), call_text))
        return value
    except BaseException as error:
        if errors:
            print_func('-> ' + _format_error(call_text, error, stack))
        raise


def print_exits(errors=True, stack=True, repr_len=REPR_LEN):
    """Prints on exits from a function."""
    if callable(errors):
        return log_exits(print)(errors)
    return log_exits(print, errors, stack, repr_len)


class LabeledContextDecorator(object):
    """
    Context manager/decorator base which supplies a call signature as a label
    when used to decorate a function.
    """

    def __init__(self, print_func, label=None, repr_len=REPR_LEN):
        self.print_func = print_func
        self.label = label
        self.repr_len = repr_len

    def __call__(self, label=None, **kwargs):
        if callable(label):
            return self.decorator(label)
        return self.__class__(self.print_func, label, **kwargs)

    def decorator(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            context = self.__class__.__new__(self.__class__)
            context.__dict__.update(self.__dict__)
            context.label = signature_repr(Call(func, args, kwargs), self.repr_len)
            with context:
                return func(*args, **kwargs)
        return wrapped


class log_errors(LabeledContextDecorator):
    """Logs or prints all errors within a function or block."""

    def __init__(self, print_func, label=None, stack=True, repr_len=REPR_LEN):
        LabeledContextDecorator.__init__(
            self, print_func, label=label, repr_len=repr_len
        )
        self.stack = stack

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type:
            if self.stack:
                message = ''.join(traceback.format_exception(exc_type, exc_value, tb))
            else:
                message = '%s: %s' % (exc_type.__name__, exc_value)
            self.print_func(_format_error(self.label, message, self.stack))


print_errors = log_errors(print)


def format_time(sec):
    if sec < 1e-6:
        return '%8.2f ns' % (sec * 1e9)
    if sec < 1e-3:
        return '%8.2f mks' % (sec * 1e6)
    if sec < 1:
        return '%8.2f ms' % (sec * 1e3)
    return '%8.2f s' % sec


time_formatters = {
    'auto': format_time,
    'ns': lambda sec: '%8.2f ns' % (sec * 1e9),
    'mks': lambda sec: '%8.2f mks' % (sec * 1e6),
    'ms': lambda sec: '%8.2f ms' % (sec * 1e3),
    's': lambda sec: '%8.2f s' % sec,
}


class log_durations(LabeledContextDecorator):
    """Times each function call or block execution."""

    def __init__(self, print_func, label=None, unit='auto', threshold=-1,
                 repr_len=REPR_LEN):
        LabeledContextDecorator.__init__(
            self, print_func, label=label, repr_len=repr_len
        )
        if unit not in time_formatters:
            raise ValueError(
                'Unknown time unit: %s. It should be ns, mks, ms, s or auto.'
                % unit
            )
        self.format_time = time_formatters[unit]
        self.threshold = threshold

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *exc):
        elapsed = timer() - self.start
        if elapsed >= self.threshold:
            text = self.format_time(elapsed)
            if self.label:
                self.print_func('%s in %s' % (text, self.label))
            else:
                self.print_func(text)


print_durations = log_durations(print)


def log_iter_durations(seq, print_func, label=None, unit='auto'):
    """Times processing of each item in seq."""
    if unit not in time_formatters:
        raise ValueError(
            'Unknown time unit: %s. It should be ns, mks, ms, s or auto.'
            % unit
        )
    render_time = time_formatters[unit]
    tail = ' of %s' % label if label else ''
    for index, value in enumerate(iter(seq)):
        started = timer()
        yield value
        elapsed = render_time(timer() - started)
        print_func('%s in iteration %d%s' % (elapsed, index, tail))


def print_iter_durations(seq, label=None, unit='auto'):
    """Times processing of each item in seq."""
    return log_iter_durations(seq, print, label, unit=unit)


def _format_error(label, e, stack=True):
    if isinstance(e, Exception):
        if stack:
            message = traceback.format_exc()
        else:
            message = '%s: %s' % (e.__class__.__name__, e)
    else:
        message = e

    if label:
        pattern = '%s    raised in %s' if stack else '%s raised in %s'
        return pattern % (message, label)
    return message


def signature_repr(call, repr_len=REPR_LEN):
    function = call._func
    if isinstance(function, partial):
        wrapped = function.func
        if hasattr(wrapped, '__name__'):
            function_name = '<%s partial>' % wrapped.__name__
        else:
            function_name = '<unknown partial>'
    else:
        function_name = getattr(function, '__name__', '<unknown>')

    positional = (smart_repr(arg, repr_len) for arg in call._args)
    named = (
        '%s=%s' % (key, smart_repr(value, repr_len))
        for key, value in call._kwargs.items()
    )
    return '%s(%s)' % (function_name, ', '.join(chain(positional, named)))


def smart_repr(value, max_len=REPR_LEN):
    if isinstance(value, (bytes, str)):
        text = repr(value)
    else:
        text = str(value)

    text = re.sub(r'\s+', ' ', text)
    if max_len and len(text) > max_len:
        text = text[:max_len - 3] + '...'
    return text