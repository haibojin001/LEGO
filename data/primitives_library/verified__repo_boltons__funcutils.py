import sys
import inspect
import functools
import itertools
import threading
import linecache

from inspect import formatannotation
from types import FunctionType, MethodType

from functools import total_ordering as total_ordering

try:
    from .typeutils import make_sentinel
    NO_DEFAULT = make_sentinel(var_name='NO_DEFAULT')
except ImportError:
    class _NoDefault:
        def __repr__(self):
            return 'NO_DEFAULT'
    NO_DEFAULT = _NoDefault()


def inspect_formatargspec(args, varargs=None, varkw=None, defaults=None,
                          kwonlyargs=(), kwonlydefaults={}, annotations={},
                          formatarg=str,
                          formatvarargs=lambda name: '*' + name,
                          formatvarkw=lambda name: '**' + name,
                          formatvalue=lambda value: '=' + repr(value),
                          formatreturns=lambda text: ' -> ' + text,
                          formatannotation=formatannotation):
    def format_one(arg):
        text = formatarg(arg)
        if arg in annotations:
            text += ': ' + formatannotation(annotations[arg])
        return text

    specs = []
    first_default = len(args) - len(defaults or ())
    for index, arg in enumerate(args):
        text = format_one(arg)
        if defaults and index >= first_default:
            text += formatvalue(defaults[index - first_default])
        specs.append(text)

    if varargs is not None:
        specs.append(formatvarargs(format_one(varargs)))
    elif kwonlyargs:
        specs.append('*')

    for arg in kwonlyargs:
        text = format_one(arg)
        if kwonlydefaults and arg in kwonlydefaults:
            text += formatvalue(kwonlydefaults[arg])
        specs.append(text)

    if varkw is not None:
        specs.append(formatvarkw(format_one(varkw)))

    result = '(' + ', '.join(specs) + ')'
    if 'return' in annotations:
        result += formatreturns(formatannotation(annotations['return']))
    return result


def get_module_callables(mod, ignore=None):
    if isinstance(mod, str):
        mod = sys.modules[mod]

    types = {}
    funcs = {}
    for name in dir(mod):
        if ignore is not None and ignore(name):
            continue
        try:
            value = getattr(mod, name)
        except Exception:
            continue
        if getattr(value, '__module__', None) != mod.__name__:
            continue
        if isinstance(value, type):
            types[name] = value
        elif callable(value):
            funcs[name] = value
    return types, funcs


def mro_items(type_obj):
    return itertools.chain.from_iterable(
        cls.__dict__.items() for cls in type_obj.__mro__
    )


def dir_dict(obj, raise_exc=False):
    ret = {}
    for name in dir(obj):
        try:
            ret[name] = getattr(obj, name)
        except Exception:
            if raise_exc:
                raise
    return ret


def copy_function(orig, copy_dict=True):
    ret = FunctionType(orig.__code__,
                       orig.__globals__,
                       name=orig.__name__,
                       argdefs=getattr(orig, '__defaults__', None),
                       closure=getattr(orig, '__closure__', None))
    if hasattr(orig, '__kwdefaults__'):
        ret.__kwdefaults__ = orig.__kwdefaults__
    if copy_dict:
        ret.__dict__.update(orig.__dict__)
    return ret


def partial_ordering(cls):
    def __lt__(self, other):
        return self <= other and not self >= other

    def __gt__(self, other):
        return self >= other and not self <= other

    def __eq__(self, other):
        return self >= other and self <= other

    if not hasattr(cls, '__lt__'):
        cls.__lt__ = __lt__
    if not hasattr(cls, '__gt__'):
        cls.__gt__ = __gt__
    if not hasattr(cls, '__eq__'):
        cls.__eq__ = __eq__
    return cls


class InstancePartial(functools.partial):
    def __new__(cls, func, *args, **kwargs):
        self = super().__new__(cls, func, *args, **kwargs)
        try:
            self.__name__ = func.__name__
        except (AttributeError, TypeError):
            pass
        return self

    def __get__(self, obj, obj_type=None):
        if obj is None:
            return self
        return MethodType(self, obj)


class CachedInstancePartial(InstancePartial):
    def __get__(self, obj, obj_type=None):
        if obj is None:
            return self

        name = getattr(self, '__name__',
                       getattr(self.func, '__name__', None))
        if name is None or not hasattr(obj, '__dict__'):
            return MethodType(self, obj)

        try:
            return obj.__dict__[name]
        except KeyError:
            bound = MethodType(self, obj)
            obj.__dict__[name] = bound
            return bound


partial = InstancePartial


def noop(*args, **kwargs):
    return None


def format_invocation(name, args=(), kwargs=None, repr=repr):
    if kwargs is None:
        kwargs = {}
    arg_text = [repr(arg) for arg in args]
    arg_text.extend('%s=%s' % (key, repr(value))
                    for key, value in kwargs.items())
    return '%s(%s)' % (name, ', '.join(arg_text))


def format_exp_repr(obj, pos_names, named_names, **kw):
    args = [getattr(obj, name) for name in pos_names]
    kwargs = dict((name, getattr(obj, name)) for name in named_names)
    kwargs.update(kw)
    return format_invocation(obj.__class__.__name__, args, kwargs)


def format_nonexp_repr(obj, named_names, **kw):
    values = dict((name, getattr(obj, name)) for name in named_names)
    values.update(kw)
    body = ' '.join('%s=%r' % (name, value)
                    for name, value in values.items())
    if body:
        return '<%s %s>' % (obj.__class__.__name__, body)
    return '<%s>' % obj.__class__.__name__


class MissingArgument(ValueError):
    pass


class ExistingArgument(ValueError):
    pass


def _indent(text, margin='    '):
    return '\n'.join(margin + line if line else line
                     for line in text.splitlines())


_builder_counter = itertools.count()
_builder_lock = threading.Lock()


class FunctionBuilder:
    _defaults = {
        'name': '_func',
        'doc': None,
        'module': None,
        'dict': None,
        'globals': None,
        'locals': None,
        'args': None,
        'varargs': None,
        'varkw': None,
        'defaults': None,
        'kwonlyargs': None,
        'kwonlydefaults': None,
        'annotations': None,
        'body': 'pass',
    }

    def __init__(self, name, **kw):
        if 'name' in kw:
            raise TypeError("FunctionBuilder() got multiple values for argument 'name'")
        values = dict(self._defaults)
        values.update(kw)
        values['name'] = name

        unexpected = set(values) - set(self._defaults)
        if unexpected:
            raise TypeError('unexpected keyword argument%s: %s'
                            % ('s' if len(unexpected) > 1 else '',
                               ', '.join(sorted(unexpected))))

        for attr, value in values.items():
            setattr(self, attr, value)

        if self.args is None:
            self.args = []
        else:
            self.args = list(self.args)

        if self.kwonlyargs is None:
            self.kwonlyargs = []
        else:
            self.kwonlyargs = list(self.kwonlyargs)

        if self.defaults is not None:
            self.defaults = tuple(self.defaults)

        if self.kwonlydefaults is not None:
            self.kwonlydefaults = dict(self.kwonlydefaults)

        if self.annotations is None:
            self.annotations = {}
        else:
            self.annotations = dict(self.annotations)

        if self.dict is None:
            self.dict = {}
        else:
            self.dict = dict(self.dict)

    @classmethod
    def from_func(cls, func):
        if isinstance(func, (classmethod, staticmethod)):
            func = func.__func__

        spec = inspect.getfullargspec(func)
        return cls(
            func.__name__,
            doc=getattr(func, '__doc__', None),
            module=getattr(func, '__module__', None),
            dict=getattr(func, '__dict__', None),
            globals=getattr(func, '__globals__', None),
            args=spec.args,
            varargs=spec.varargs,
            varkw=spec.varkw,
            defaults=spec.defaults,
            kwonlyargs=spec.kwonlyargs,
            kwonlydefaults=spec.kwonlydefaults,
            annotations=spec.annotations,
        )

    def get_sig_str(self, with_annotations=True):
        annotations = self.annotations if with_annotations else {}
        return inspect_formatargspec(
            self.args,
            self.varargs,
            self.varkw,
            self.defaults,
            self.kwonlyargs,
            self.kwonlydefaults,
            annotations,
        )

    def get_invocation_str(self):
        parts = list(self.args)
        if self.varargs:
            parts.append('*' + self.varargs)
        parts.extend('%s=%s' % (name, name) for name in self.kwonlyargs)
        if self.varkw:
            parts.append('**' + self.varkw)
        return ', '.join(parts)

    def get_defaults_dict(self):
        defaults = self.defaults or ()
        if not defaults:
            return {}
        return dict(zip(self.args[-len(defaults):], defaults))

    def get_arg_names(self, only_required=False):
        ret = list(self.args)
        ret.extend(self.kwonlyargs)
        if only_required:
            defaults = self.get_defaults_dict()
            ret = [name for name in ret
                   if name not in defaults
                   and name not in (self.kwonlydefaults or {})]
        return ret

    def add_arg(self, arg_name, default=NO_DEFAULT, kwonly=False):
        if arg_name in self.args or arg_name in self.kwonlyargs:
            raise ExistingArgument(arg_name)

        if kwonly:
            self.kwonlyargs.append(arg_name)
            if default is not NO_DEFAULT:
                if self.kwonlydefaults is None:
                    self.kwonlydefaults = {}
                self.kwonlydefaults[arg_name] = default
            return

        if default is NO_DEFAULT:
            defaults = self.defaults or ()
            if defaults:
                self.args.insert(len(self.args) - len(defaults), arg_name)
            else:
                self.args.append(arg_name)
            return

        self.args.append(arg_name)
        self.defaults = (self.defaults or ()) + (default,)

    def remove_arg(self, arg_name):
        if arg_name in self.kwonlyargs:
            self.kwonlyargs.remove(arg_name)
            if self.kwonlydefaults:
                self.kwonlydefaults.pop(arg_name, None)
            return

        if arg_name not in self.args:
            raise MissingArgument(arg_name)

        defaults = self.defaults or ()
        first_default = len(self.args) - len(defaults)
        index = self.args.index(arg_name)
        self.args.pop(index)

        if index >= first_default:
            default_index = index - first_default
            new_defaults = list(defaults)
            new_defaults.pop(default_index)
            self.defaults = tuple(new_defaults) or None

    def get_func(self, execdict=None, add_source=True, with_dict=True):
        if execdict is None:
            execdict = {}
            if self.globals:
                execdict.update(self.globals)
        else:
            execdict = dict(execdict)

        signature = self.get_sig_str()
        source = 'def %s%s:\n%s\n' % (
            self.name,
            signature,
            _indent(self.body or 'pass'),
        )

        with _builder_lock:
            number = next(_builder_counter)
        filename = '<boltons.funcutils.FunctionBuilder-%d>' % number
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )

        code = compile(source, filename, 'exec')
        local_ns = {}
        exec(code, execdict, local_ns)
        func = local_ns[self.name]

        func.__doc__ = self.doc
        func.__module__ = self.module
        if with_dict:
            func.__dict__.update(self.dict)
        if add_source:
            func.__source__ = source
        return func


def _parse_wraps_expected(expected):
    if isinstance(expected, str):
        expected = expected.replace(',', ' ').split()

    try:
        expected = list(expected)
    except TypeError:
        raise ValueError('expected must be a string or iterable')

    ret = []
    for item in expected:
        if isinstance(item, str):
            ret.append((item, NO_DEFAULT))
            continue
        try:
            name, default = item
        except (TypeError, ValueError):
            raise ValueError(
                'expected items must be argument names or (name, default) pairs'
            )
        ret.append((name, default))
    return ret


def update_wrapper(wrapper, func, injected=None, expected=None,
                   build_from=None, **kw):
    if isinstance(func, (classmethod, staticmethod)):
        func = func.__func__
    if isinstance(wrapper, (classmethod, staticmethod)):
        wrapper = wrapper.__func__

    functools.update_wrapper(wrapper, func, **kw)

    if build_from is None:
        build_from = func
    elif build_from == 'func':
        build_from = func
    elif build_from == 'wrapper':
        build_from = wrapper
    elif not callable(build_from):
        raise ValueError('build_from must be "func", "wrapper", or a callable')

    builder = FunctionBuilder.from_func(build_from)

    if injected:
        for arg_name in injected:
            try:
                builder.remove_arg(arg_name)
            except MissingArgument:
                raise ValueError(
                    'arg %r not found in function signature' % arg_name
                )

    if expected:
        for arg_name, default in _parse_wraps_expected(expected):
            builder.add_arg(arg_name, default)

    builder.name = wrapper.__name__
    builder.doc = wrapper.__doc__
    builder.module = wrapper.__module__
    builder.dict = dict(wrapper.__dict__)
    builder.body = 'return _call(%s)' % builder.get_invocation_str()

    return builder.get_func(
        execdict={'_call': wrapper},
        add_source=True,
        with_dict=True,
    )


def wraps(func, injected=None, expected=None, **kw):
    return functools.partial(
        update_wrapper,
        func=func,
        injected=injected,
        expected=expected,
        **kw
    )