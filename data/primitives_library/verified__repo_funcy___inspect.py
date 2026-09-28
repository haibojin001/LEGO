from __future__ import absolute_import

from collections import namedtuple
from inspect import CO_VARARGS, CO_VARKEYWORDS, signature
import platform
import re

from .decorators import unwrap


IS_PYPY = platform.python_implementation() == "PyPy"


ARGS = {
    'builtins': {
        'bool': '*',
        'complex': 'real,imag',
        'enumerate': 'iterable,start',
        'file': 'file-**',
        'float': 'x',
        'int': 'x-*',
        'long': 'x-*',
        'open': 'file-**',
        'round': 'number-*',
        'setattr': '***',
        'str': 'object-*',
        'unicode': 'string-**',
        '__import__': 'name-****',
        '__buildclass__': '***',
        'iter': '*-*',
        'format': '*-*',
        'type': '*-**',
    },
    'functools': {
        'reduce': '**',
    },
    'itertools': {
        'accumulate': 'iterable-*',
        'combinations': 'iterable,r',
        'combinations_with_replacement': 'iterable,r',
        'compress': 'data,selectors',
        'groupby': 'iterable-*',
        'permutations': 'iterable-*',
        'repeat': 'object-*',
    },
    'operator': {
        'delslice': '***',
        'getslice': '***',
        'setitem': '***',
        'setslice': '****',
    },
    'funcy.seqs': {
        'map': 'f*',
        'lmap': 'f*',
        'xmap': 'f*',
        'mapcat': 'f*',
        'lmapcat': 'f*',
    },
    'funcy.colls': {
        'merge_with': 'f*',
    },
}

ARGS['builtins'].update(dict.fromkeys(
    'cmp coerce delattr divmod filter getattr hasattr isinstance issubclass '
    'map pow reduce'.split(),
    '**'
))

ARGS['itertools'].update(dict.fromkeys(
    'dropwhile filterfalse ifilter ifilterfalse starmap takewhile'.split(),
    '**'
))

_operator_two_arg = """
    _compare_digest add and_ concat contains countOf delitem div eq floordiv ge getitem
    gt iadd iand iconcat idiv ifloordiv ilshift imatmul imod imul indexOf ior ipow irepeat
    irshift is_ is_not isub itruediv ixor le lshift lt matmul mod mul ne or_ pow repeat rshift
    sequenceIncludes sub truediv xor
"""
ARGS['operator'].update(dict.fromkeys(_operator_two_arg.split(), '**'))
ARGS['operator'].update(
    ('__%s__' % name.strip('_'), value)
    for name, value in list(ARGS['operator'].items())
)
ARGS['_operator'] = ARGS['operator']

STD_MODULES = set(ARGS)

Spec = namedtuple('Spec', 'max_n names req_n req_names varkw')


def _from_known_description(description):
    mandatory, separator, extra = description.partition('-')
    mandatory_items = re.findall(r'\w+|\*', mandatory)
    return Spec(
        max_n=len(mandatory_items) + len(extra),
        names=set(),
        req_n=len(mandatory_items),
        req_names=set(mandatory_items),
        varkw=False,
    )


def get_spec(func, _cache={}):
    func = getattr(func, '__original__', None) or unwrap(func)

    try:
        return _cache[func]
    except (KeyError, TypeError):
        pass

    module_name = getattr(func, '__module__', None)

    if module_name in STD_MODULES or (
        module_name in ARGS and func.__name__ in ARGS[module_name]
    ):
        result = _from_known_description(
            ARGS[module_name].get(func.__name__, '*')
        )
        _cache[func] = result
        return result

    if isinstance(func, type):
        inherited_from = getattr(func.__init__, '__objclass__', None)
        if inherited_from and inherited_from is not func:
            return get_spec(inherited_from)

        result = get_spec(func.__init__)
        instance_name = func.__init__.__code__.co_varnames[0]
        remove = {instance_name}
        return result._replace(
            max_n=result.max_n - 1,
            names=result.names - remove,
            req_n=result.req_n - 1,
            req_names=result.req_names - remove,
        )

    if not IS_PYPY and hasattr(func, '__code__'):
        return _code_to_spec(func)

    try:
        inspected = signature(func)
    except (ValueError, TypeError):
        label = getattr(func, '__qualname__', None) or getattr(
            func, '__name__', func
        )
        raise ValueError('Unable to introspect %s() arguments' % label)
    else:
        result = _sig_to_spec(inspected)
        _cache[func] = result
        return result


def _code_to_spec(func):
    code = func.__code__

    defaults = getattr(func, '__defaults__', None)
    default_count = len(defaults) if isinstance(defaults, tuple) else 0

    keyword_defaults = getattr(func, '__kwdefaults__', None)
    if not isinstance(keyword_defaults, dict):
        keyword_defaults = {}

    positional_only = getattr(code, 'co_posonlyargcount', 0)
    positional_count = code.co_argcount
    total_count = positional_count + code.co_kwonlyargcount
    variables = code.co_varnames

    accepted_names = set(variables[positional_only:total_count])
    required_count = total_count - default_count - len(keyword_defaults)
    required_names = (
        set(
            variables[positional_only:positional_count - default_count]
            + variables[positional_count:total_count]
        )
        - set(keyword_defaults)
    )

    accepts_keywords = bool(code.co_flags & CO_VARKEYWORDS)
    maximum_count = total_count + bool(code.co_flags & CO_VARARGS)

    return Spec(
        max_n=maximum_count,
        names=accepted_names,
        req_n=required_count,
        req_names=required_names,
        varkw=accepts_keywords,
    )


def _sig_to_spec(sig):
    maximum_count = 0
    accepted_names = set()
    required_count = 0
    required_names = set()
    accepts_keywords = False

    for name, parameter in sig.parameters.items():
        maximum_count += 1

        if parameter.kind == parameter.VAR_KEYWORD:
            maximum_count -= 1
            accepts_keywords = True
        elif parameter.kind == parameter.VAR_POSITIONAL:
            required_count += 1
        elif parameter.kind == parameter.POSITIONAL_ONLY:
            if parameter.default is parameter.empty:
                required_count += 1
        else:
            accepted_names.add(name)
            if parameter.default is parameter.empty:
                required_count += 1
                required_names.add(name)

    return Spec(
        max_n=maximum_count,
        names=accepted_names,
        req_n=required_count,
        req_names=required_names,
        varkw=accepts_keywords,
    )