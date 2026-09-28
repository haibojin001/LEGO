import sys
import time
from reprlib import Repr

try:
    from .typeutils import make_sentinel
    _UNSET = make_sentinel(var_name='_UNSET')
except ImportError:
    _UNSET = object()

__all__ = ['pdb_on_signal', 'pdb_on_exception', 'wrap_trace']


def pdb_on_signal(signalnum=None):
    import pdb
    import signal

    if not signalnum:
        signalnum = signal.SIGINT

    prior_handler = signal.getsignal(signalnum)

    def handler(sig, frame):
        signal.signal(signalnum, prior_handler)
        pdb.set_trace()
        pdb_on_signal(signalnum)

    signal.signal(signalnum, handler)
    return None


def pdb_on_exception(limit=100):
    import pdb
    import traceback

    def exception_hook(exc_type, exc_value, exc_traceback):
        traceback.print_tb(exc_traceback, limit=limit)
        pdb.post_mortem(exc_traceback)

    sys.excepthook = exception_hook
    return None


_repr = Repr()
_repr.maxstring = 50
_repr.maxother = 50
brief_repr = _repr.repr


def trace_print_hook(event, label, obj, attr_name,
                     args=(), kwargs={}, result=_UNSET):
    values = (
        event.ljust(6),
        time.time(),
        label.rjust(10),
        obj.__class__.__name__,
        attr_name,
    )

    if event == 'get':
        template = '%s %s - %s - %s.%s -> %s'
        values += (brief_repr(result),)
    elif event == 'set':
        template = '%s %s - %s - %s.%s = %s'
        values += (brief_repr(args[0]),)
    elif event == 'del':
        template = '%s %s - %s - %s.%s'
    else:
        template = '%s %s - %s - %s.%s(%s)'
        values += (', '.join(brief_repr(arg) for arg in args),)
        if kwargs:
            template = '%s %s - %s - %s.%s(%s, %s)'
            values += (
                ', '.join('%s=%s' % (key, brief_repr(value))
                          for key, value in kwargs.items()),
            )
        if result is not _UNSET:
            template += ' -> %s'
            values += (brief_repr(result),)

    print(template % values)
    return None


def wrap_trace(obj, hook=trace_print_hook,
               which=None, events=None, label=None):
    if isinstance(which, str):
        selector = lambda name, value: name == which
    elif callable(getattr(which, '__contains__', None)):
        selector = lambda name, value: name in which
    elif which is None or callable(which):
        selector = which
    else:
        raise TypeError('expected attr name(s) or callable, not: %r' % which)

    if not label:
        label = hex(id(obj))

    if isinstance(events, str):
        events = [events]

    trace_get = not events or 'get' in events
    trace_set = not events or 'set' in events
    trace_del = not events or 'del' in events
    trace_call = not events or 'call' in events
    trace_raise = not events or 'raise' in events
    trace_return = not events or 'return' in events

    def method_wrapper(name, method, _hook=hook, _label=label):
        def traced_method(*args, **kwargs):
            args = args[1:]

            if trace_call:
                hook(event='call', label=_label, obj=obj,
                     attr_name=name, args=args, kwargs=kwargs)

            if trace_raise:
                try:
                    result = method(*args, **kwargs)
                except Exception:
                    if not hook(event='raise', label=_label, obj=obj,
                                attr_name=name, args=args, kwargs=kwargs,
                                result=sys.exc_info()):
                        raise
            else:
                result = method(*args, **kwargs)

            if trace_return:
                hook(event='return', label=_label, obj=obj,
                     attr_name=name, args=args, kwargs=kwargs,
                     result=result)
            return result

        traced_method.__name__ = method.__name__
        traced_method.__doc__ = method.__doc__
        try:
            traced_method.__module__ = method.__module__
        except Exception:
            pass
        try:
            if method.__dict__:
                traced_method.__dict__.update(method.__dict__)
        except Exception:
            pass
        return traced_method

    def __getattribute__(self, attr_name):
        result = type(obj).__getattribute__(obj, attr_name)
        if callable(result):
            result = type(obj).__getattribute__(self, attr_name)
        if trace_get:
            hook('get', label, obj, attr_name, (), {}, result=result)
        return result

    def __setattr__(self, attr_name, value):
        type(obj).__setattr__(obj, attr_name, value)
        if trace_set:
            hook('set', label, obj, attr_name, (value,), {})
        return None

    def __delattr__(self, attr_name):
        type(obj).__delattr__(obj, attr_name)
        if trace_del:
            hook('del', label, obj, attr_name, (), {})
        return None

    attrs = {}
    for attr_name in dir(obj):
        try:
            attr_value = getattr(obj, attr_name)
        except Exception:
            continue
        if callable(attr_value):
            if selector is None or selector(attr_name, attr_value):
                attrs[attr_name] = method_wrapper(attr_name, attr_value)

    attrs.update({
        '__getattribute__': __getattribute__,
        '__setattr__': __setattr__,
        '__delattr__': __delattr__,
    })

    trace_type = type('Traced%s' % obj.__class__.__name__,
                      (obj.__class__,), attrs)
    return trace_type()