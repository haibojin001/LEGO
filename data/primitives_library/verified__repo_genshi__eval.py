# -*- coding: utf-8 -*-
"""Support for controlled evaluation of Python expressions in templates."""

from textwrap import dedent
from types import CodeType

from genshi.compat import builtins, exec_, string_types, text_type
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten

from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
    build_code_chunk, isstring, IS_PYTHON2, _ast_Str


__all__ = [
    'Code', 'Expression', 'Suite', 'LenientLookup', 'StrictLookup',
    'Undefined', 'UndefinedError'
]

__docformat__ = 'restructuredtext en'


def _parse(source, mode):
    return parse(dedent(source), mode=mode)


def _name_node(name, ctx=None):
    if ctx is None:
        ctx = _ast.Load()
    return _ast.Name(id=name, ctx=ctx)


def _string_node(value):
    if hasattr(_ast, 'Constant'):
        return _ast.Constant(value=value)
    return _ast.Str(s=value)


def _call_node(function, arguments):
    return _ast.Call(func=_name_node(function), args=arguments, keywords=[])


def _copy_location(new_node, old_node):
    try:
        return _ast.copy_location(new_node, old_node)
    except AttributeError:
        return new_node


class _LookupTransformer(ASTTransformer):

    _protected_names = frozenset((
        '__data__', '_lookup_name', '_lookup_attr', '_lookup_item'
    ))

    def __init__(self):
        ASTTransformer.__init__(self)
        self._local_names = []

    def _is_local(self, name):
        return any(name in scope for scope in reversed(self._local_names))

    def _push_scope(self, names):
        self._local_names.append(set(names))

    def _pop_scope(self):
        self._local_names.pop()

    def _target_names(self, node):
        names = set()

        def collect(value):
            if isinstance(value, _ast.Name):
                names.add(value.id)
            elif isinstance(value, (getattr(_ast, 'Tuple', ()),
                                    getattr(_ast, 'List', ()))):
                for item in value.elts:
                    collect(item)
            elif isinstance(value, getattr(_ast, 'Starred', ())):
                collect(value.value)

        collect(node)
        return names

    def visit_Name(self, node):
        if not isinstance(node.ctx, _ast.Load):
            return node
        if node.id in self._protected_names or self._is_local(node.id):
            return node
        replacement = _call_node('_lookup_name', [
            _name_node('__data__'),
            _string_node(node.id)
        ])
        return _copy_location(replacement, node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.ctx, _ast.Load):
            return node
        replacement = _call_node('_lookup_attr', [
            node.value,
            _string_node(node.attr)
        ])
        return _copy_location(replacement, node)

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.ctx, _ast.Load):
            return node
        replacement = _call_node('_lookup_item', [
            node.value,
            node.slice
        ])
        return _copy_location(replacement, node)

    def visit_Lambda(self, node):
        defaults = []
        for default in node.args.defaults:
            defaults.append(self.visit(default))
        node.args.defaults = defaults
        if getattr(node.args, 'kw_defaults', None) is not None:
            node.args.kw_defaults = [
                self.visit(default) if default is not None else None
                for default in node.args.kw_defaults
            ]

        names = set()
        for argument in list(getattr(node.args, 'posonlyargs', ())) + \
                        list(node.args.args) + \
                        list(node.args.kwonlyargs):
            names.add(argument.arg)
        if node.args.vararg is not None:
            names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            names.add(node.args.kwarg.arg)

        self._push_scope(names)
        try:
            node.body = self.visit(node.body)
        finally:
            self._pop_scope()
        return node

    def _visit_comprehension_expression(self, node):
        generators = node.generators
        self._push_scope(set())
        try:
            for generator in generators:
                generator.iter = self.visit(generator.iter)
                self._local_names[-1].update(self._target_names(generator.target))
                generator.ifs = [self.visit(condition)
                                 for condition in generator.ifs]
            if hasattr(node, 'elt'):
                node.elt = self.visit(node.elt)
            if hasattr(node, 'key'):
                node.key = self.visit(node.key)
            if hasattr(node, 'value'):
                node.value = self.visit(node.value)
        finally:
            self._pop_scope()
        return node

    def visit_ListComp(self, node):
        return self._visit_comprehension_expression(node)

    def visit_SetComp(self, node):
        return self._visit_comprehension_expression(node)

    def visit_GeneratorExp(self, node):
        return self._visit_comprehension_expression(node)

    def visit_DictComp(self, node):
        return self._visit_comprehension_expression(node)


def _compile(node, source, mode='eval', filename=None, lineno=-1, xform=None):
    if xform is None:
        xform = _LookupTransformer

    if isinstance(xform, type):
        transformer = xform()
    else:
        transformer = xform

    node = transformer.visit(node)

    if lineno is not None and lineno != -1:
        try:
            node = _ast.increment_lineno(node, lineno - 1)
        except AttributeError:
            pass

    try:
        node = _ast.fix_missing_locations(node)
    except AttributeError:
        pass

    filename = filename or '<string>'

    if IS_PYTHON2:
        generator = ASTCodeGenerator()
        source_code = generator.visit(node)
        return compile(source_code, filename, mode)

    return compile(node, filename, mode)


class Code(object):
    """Base class shared by compiled template expressions and suites."""

    __slots__ = ['source', 'code', 'ast', '_globals']

    def __init__(self, source, filename=None, lineno=-1, lookup='strict',
                 xform=None):
        if isinstance(source, string_types):
            self.source = source
            node = _parse(source, mode=self.mode)
        else:
            assert isinstance(source, _ast.AST), \
                'Expected string or AST node, but got %r' % source
            self.source = '?'
            if self.mode == 'eval':
                node = _ast.Expression(body=source)
            else:
                try:
                    node = _ast.Module(body=[source], type_ignores=[])
                except TypeError:
                    node = _ast.Module()
                    node.body = [source]

        self.ast = node
        self.code = _compile(node, self.source, mode=self.mode,
                             filename=filename, lineno=lineno, xform=xform)

        if lookup is None:
            lookup = LenientLookup
        elif isinstance(lookup, string_types):
            lookup = {
                'lenient': LenientLookup,
                'strict': StrictLookup
            }[lookup]

        self._globals = lookup.globals

    def __getstate__(self):
        if hasattr(self._globals, '__self__'):
            lookup = self._globals.__self__
        else:
            lookup = self._globals.im_self
        return {
            'source': self.source,
            'ast': self.ast,
            'lookup': lookup,
            'code': get_code_params(self.code)
        }

    def __setstate__(self, state):
        self.source = state['source']
        self.ast = state['ast']
        self.code = CodeType(0, *state['code'])
        self._globals = state['lookup'].globals

    def __eq__(self, other):
        return type(self) is type(other) and self.code == other.code

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.source)


class Expression(Code):
    """A compiled Python expression evaluated against template data."""

    __slots__ = []

    mode = 'eval'

    def evaluate(self, data):
        __traceback_hide__ = 'before_and_this'
        return eval(self.code, self._globals(data), {'__data__': data})


class Suite(Code):
    """A compiled Python suite executed against template data."""

    __slots__ = []

    mode = 'exec'

    def execute(self, data):
        __traceback_hide__ = 'before_and_this'
        exec_(self.code, self._globals(data), data)


UNDEFINED = object()


class UndefinedError(TemplateRuntimeError):
    """Raised when an unavailable template value is used."""

    def __init__(self, name, owner=UNDEFINED):
        if owner is UNDEFINED:
            message = '"%s" not defined' % name
        else:
            message = '%s has no member named "%s"' % (repr(owner), name)
        TemplateRuntimeError.__init__(self, message)
        if not hasattr(self, 'msg'):
            self.msg = message


class Undefined(object):
    """A false, empty placeholder used by lenient lookups."""

    __slots__ = ['_name', '_owner']

    def __init__(self, name, owner=UNDEFINED):
        self._name = name
        self._owner = owner

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    __nonzero__ = __bool__

    def __repr__(self):
        return '<%s %r>' % (type(self).__name__, self._name)

    def __str__(self):
        return 'undefined'

    def _die(self, *args, **kwargs):
        __traceback_hide__ = True
        raise UndefinedError(self._name, self._owner)

    __call__ = _die
    __getattr__ = _die
    __getitem__ = _die

    __length_hint__ = None


class LookupBase(object):
    """Base implementation for expression name, attribute, and item lookup."""

    @classmethod
    def globals(cls, data):
        return {
            '__data__': data,
            '__builtins__': builtins,
            '_lookup_name': cls.lookup_name,
            '_lookup_attr': cls.lookup_attr,
            '_lookup_item': cls.lookup_item
        }

    @staticmethod
    def _builtin(name):
        try:
            return builtins[name]
        except (TypeError, KeyError):
            try:
                return getattr(builtins, name)
            except AttributeError:
                return UNDEFINED

    @classmethod
    def lookup_name(cls, data, key):
        raise NotImplementedError

    @classmethod
    def lookup_attr(cls, obj, key):
        raise NotImplementedError

    @classmethod
    def lookup_item(cls, obj, key):
        raise NotImplementedError


class LenientLookup(LookupBase):
    """Lookup policy that yields :class:`Undefined` for missing values."""

    @classmethod
    def lookup_name(cls, data, key):
        try:
            return data[key]
        except (KeyError, TypeError, IndexError):
            value = cls._builtin(key)
            if value is not UNDEFINED:
                return value
            return Undefined(key)

    @classmethod
    def lookup_attr(cls, obj, key):
        try:
            return obj[key]
        except (KeyError, TypeError, IndexError, AttributeError):
            try:
                return getattr(obj, key)
            except (AttributeError, TypeError):
                return Undefined(key, obj)

    @classmethod
    def lookup_item(cls, obj, key):
        try:
            return obj[key]
        except (KeyError, TypeError, IndexError, AttributeError):
            if isinstance(key, string_types):
                try:
                    return getattr(obj, key)
                except (AttributeError, TypeError):
                    pass
            return Undefined(key, obj)


class StrictLookup(LookupBase):
    """Lookup policy that raises :class:`UndefinedError` for missing values."""

    @classmethod
    def lookup_name(cls, data, key):
        try:
            return data[key]
        except (KeyError, TypeError, IndexError):
            value = cls._builtin(key)
            if value is not UNDEFINED:
                return value
            raise UndefinedError(key)

    @classmethod
    def lookup_attr(cls, obj, key):
        try:
            return obj[key]
        except (KeyError, TypeError, IndexError, AttributeError):
            try:
                return getattr(obj, key)
            except (AttributeError, TypeError):
                raise UndefinedError(key, obj)

    @classmethod
    def lookup_item(cls, obj, key):
        try:
            return obj[key]
        except (KeyError, TypeError, IndexError, AttributeError):
            if isinstance(key, string_types):
                try:
                    return getattr(obj, key)
                except (AttributeError, TypeError):
                    pass
            raise UndefinedError(key, obj)