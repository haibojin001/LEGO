import copy
import logging
import operator
import time
from collections import deque
from io import StringIO

import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)
np_versions = list(map(int, np.__version__.split(".")[:2]))

DEFAULT_MAX_PROGRAM_LEN = 100000


class PyllImportError(ImportError):
    """Raised when a requested pyll symbol does not exist."""


class MissingArgument:
    """Marker used for omitted arguments in symbolic calls."""


class SymbolTable:
    """Registry of symbolic operations and their Python implementations."""

    def __init__(self):
        self._impls = {
            "list": list,
            "dict": dict,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "map": map,
            "max": max,
            "min": min,
            "getattr": getattr,
        }

    def _new_apply(self, name, args, kwargs, o_len, pure):
        pos_args = [as_apply(arg) for arg in args]
        named_args = [(key, as_apply(value)) for key, value in kwargs.items()]
        named_args.sort()
        return Apply(name, pos_args, named_args, o_len=o_len, pure=pure)

    def dict(self, *args, **kwargs):
        return self._new_apply("dict", args, kwargs, o_len=None, pure=True)

    def int(self, arg):
        return self._new_apply("int", [arg], {}, o_len=None, pure=True)

    def float(self, arg):
        return self._new_apply("float", [arg], {}, o_len=None, pure=True)

    def len(self, obj):
        return self._new_apply("len", [obj], {}, o_len=None, pure=True)

    def list(self, init):
        return self._new_apply("list", [init], {}, o_len=None, pure=True)

    def map(self, fn, seq, pure=False):
        return self._new_apply("map", [fn, seq], {}, o_len=seq.o_len, pure=pure)

    def range(self, *args):
        return self._new_apply("range", args, {}, o_len=None, pure=True)

    def max(self, *args):
        return self._new_apply("max", args, {}, o_len=None, pure=True)

    def min(self, *args):
        return self._new_apply("min", args, {}, o_len=None, pure=True)

    def getattr(self, obj, attr, *args):
        return self._new_apply(
            "getattr", [obj, attr] + list(args), {}, o_len=None, pure=True
        )

    def _define(self, f, o_len, pure):
        name = f.__name__
        entry = SymbolTableEntry(self, name, o_len, pure)
        setattr(self, name, entry)
        self._impls[name] = f
        return f

    def define(self, f, o_len=None, pure=False):
        name = f.__name__
        if hasattr(self, name):
            raise ValueError("Cannot override existing symbol", name)
        return self._define(f, o_len, pure)

    def define_if_new(self, f, o_len=None, pure=False):
        name = f.__name__
        if hasattr(self, name) and self._impls[name] is not f:
            raise ValueError("Cannot redefine existing symbol", name)
        return self._define(f, o_len, pure)

    def undefine(self, f):
        name = f if isinstance(f, str) else f.__name__
        del self._impls[name]
        delattr(self, name)

    def define_pure(self, f):
        return self.define(f, o_len=None, pure=True)

    def define_info(self, o_len=None, pure=False):
        def decorator(f):
            return self.define(f, o_len=o_len, pure=pure)

        return decorator

    def inject(self, *args, **kwargs):
        result = {}
        for name in args:
            try:
                result[name] = getattr(self, name)
            except AttributeError:
                raise PyllImportError(name)
        for alias, name in kwargs.items():
            try:
                result[alias] = getattr(self, name)
            except AttributeError:
                raise PyllImportError(name)
        return result

    def import_(self, _globals, *args, **kwargs):
        _globals.update(self.inject(*args, **kwargs))


class SymbolTableEntry:
    """Callable proxy which creates an Apply node for a symbol table entry."""

    def __init__(self, symbol_table, apply_name, o_len, pure):
        self.symbol_table = symbol_table
        self.apply_name = apply_name
        self.o_len = o_len
        self.pure = pure

    def __call__(self, *args, **kwargs):
        return self.symbol_table._new_apply(
            self.apply_name, args, kwargs, self.o_len, self.pure
        )


scope = SymbolTable()


def as_apply(obj):
    """Convert a Python object or symbolic node to an Apply instance."""
    if isinstance(obj, Apply):
        result = obj
    elif isinstance(obj, tuple):
        result = Apply(
            "pos_args",
            [as_apply(item) for item in obj],
            [],
            o_len=len(obj),
            pure=True,
        )
    elif isinstance(obj, list):
        result = Apply(
            "pos_args",
            [as_apply(item) for item in obj],
            [],
            o_len=None,
            pure=True,
        )
    elif isinstance(obj, dict):
        items = list(obj.items())
        items.sort()
        if all(isinstance(key, str) for key in obj):
            result = Apply(
                "dict",
                [],
                [(key, as_apply(value)) for key, value in items],
                o_len=len(items),
                pure=True,
            )
        else:
            result = Apply(
                "dict",
                [as_apply([(key, as_apply(value)) for key, value in items])],
                [],
                o_len=None,
                pure=True,
            )
    else:
        result = Literal(obj)
    assert isinstance(result, Apply)
    return result


class Apply:
    """A symbolic application node."""

    def __init__(
        self, name, pos_args, named_args, o_len=None, pure=False, define_params=None
    ):
        self.name = name
        self.pos_args = list(pos_args)
        self.named_args = [[key, value] for key, value in named_args]
        self.o_len = o_len
        self.pure = pure
        self.define_params = define_params
        assert all(isinstance(arg, Apply) for arg in self.pos_args)
        assert all(isinstance(value, Apply) for _, value in self.named_args)
        assert all(isinstance(key, str) for key, _ in self.named_args)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.define_params:
            scope.define_if_new(**self.define_params)

    def inputs(self):
        return self.pos_args + [value for _, value in self.named_args]

    def replace_input(self, old_node, new_node):
        new_node = as_apply(new_node)
        self.pos_args = [
            new_node if arg is old_node else arg for arg in self.pos_args
        ]
        self.named_args = [
            [key, new_node if value is old_node else value]
            for key, value in self.named_args
        ]
        return self

    def eval(self, memo=None):
        if memo is None:
            memo = {}
        node_id = id(self)
        if node_id in memo:
            return memo[node_id]
        args = [arg.eval(memo) for arg in self.pos_args]
        kwargs = {key: value.eval(memo) for key, value in self.named_args}
        try:
            fn = scope._impls[self.name]
        except KeyError:
            raise KeyError("No implementation for symbol %r" % self.name)
        value = fn(*args, **kwargs)
        memo[node_id] = value
        return value

    def __len__(self):
        if self.o_len is None:
            raise TypeError("length of symbolic object is not known")
        return self.o_len

    def __iter__(self):
        if self.o_len is None:
            raise TypeError("symbolic object is not iterable")
        for index in range(self.o_len):
            yield self[index]

    def __getitem__(self, item):
        return scope.getitem(self, item)

    def __call__(self, *args, **kwargs):
        return scope.call(self, *args, **kwargs)

    def pprint(self, ofile=None, indent=0, memo=None):
        own_file = ofile is None
        if own_file:
            ofile = StringIO()
        if memo is None:
            memo = {}

        def render(node, level):
            prefix = " " * level
            node_id = id(node)
            if node_id in memo:
                return prefix + "<%d>" % memo[node_id]
            memo[node_id] = len(memo)
            if isinstance(node, Literal):
                return prefix + repr(node.obj)
            args = [render(arg, level + 2).lstrip() for arg in node.pos_args]
            args.extend(
                "%s=%s" % (key, render(value, level + 2).lstrip())
                for key, value in node.named_args
            )
            if not args:
                return prefix + node.name + "()"
            return prefix + node.name + "(" + ", ".join(args) + ")"

        ofile.write(render(self, indent))
        if own_file:
            return ofile.getvalue()
        return None

    def __str__(self):
        return self.pprint()

    def __repr__(self):
        return self.pprint()

    def __add__(self, other):
        return scope.add(self, other)

    def __radd__(self, other):
        return scope.add(other, self)

    def __sub__(self, other):
        return scope.sub(self, other)

    def __rsub__(self, other):
        return scope.sub(other, self)

    def __mul__(self, other):
        return scope.mul(self, other)

    def __rmul__(self, other):
        return scope.mul(other, self)

    def __truediv__(self, other):
        return scope.truediv(self, other)

    def __rtruediv__(self, other):
        return scope.truediv(other, self)

    def __floordiv__(self, other):
        return scope.floordiv(self, other)

    def __rfloordiv__(self, other):
        return scope.floordiv(other, self)

    def __mod__(self, other):
        return scope.mod(self, other)

    def __rmod__(self, other):
        return scope.mod(other, self)

    def __pow__(self, other):
        return scope.pow(self, other)

    def __rpow__(self, other):
        return scope.pow(other, self)

    def __neg__(self):
        return scope.neg(self)

    def __pos__(self):
        return scope.pos(self)

    def __abs__(self):
        return scope.abs(self)

    def __invert__(self):
        return scope.invert(self)

    def __lt__(self, other):
        return scope.lt(self, other)

    def __le__(self, other):
        return scope.le(self, other)

    def __gt__(self, other):
        return scope.gt(self, other)

    def __ge__(self, other):
        return scope.ge(self, other)


class Literal(Apply):
    """An Apply node whose value is a Python object."""

    def __init__(self, obj):
        self.obj = obj
        super().__init__("literal", [], [], o_len=None, pure=True)

    def eval(self, memo=None):
        return self.obj

    def __str__(self):
        return repr(self.obj)

    def __repr__(self):
        return "Literal{%r}" % (self.obj,)

    def pprint(self, ofile=None, indent=0, memo=None):
        if ofile is None:
            return " " * indent + repr(self.obj)
        ofile.write(" " * indent + repr(self.obj))
        return None


class Lambda:
    """A small symbolic lambda object."""

    def __init__(self, expr, params):
        self.expr = as_apply(expr)
        self.params = tuple(as_apply(param) for param in params)

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError("Lambda does not accept keyword arguments")
        if len(args) != len(self.params):
            raise TypeError(
                "Lambda expected %d arguments, got %d" % (len(self.params), len(args))
            )
        replacements = dict(zip(self.params, map(as_apply, args)))
        return clone(self.expr, replace=replacements)


def dfs(expr, seq=None, memo=None):
    """Return nodes in dependency-first depth-first order."""
    expr = as_apply(expr)
    if seq is None:
        seq = []
    if memo is None:
        memo = set()

    def visit(node):
        node_id = id(node)
        if node_id in memo:
            return
        memo.add(node_id)
        for child in node.inputs():
            visit(child)
        seq.append(node)

    visit(expr)
    return seq


def toposort(expr):
    """Return a topological ordering of the expression graph."""
    return dfs(expr)


def pyll_dfs(expr, seq=None, memo=None):
    return dfs(expr, seq=seq, memo=memo)


def clone(expr, memo=None, replace=None):
    """Copy an expression graph while preserving shared subexpressions."""
    expr = as_apply(expr)
    if memo is None:
        memo = {}
    if replace is None:
        replace = {}

    replacement_ids = {id(key): value for key, value in replace.items()}

    def visit(node):
        node_id = id(node)
        if node_id in replacement_ids:
            return as_apply(replacement_ids[node_id])
        if node_id in memo:
            return memo[node_id]
        if isinstance(node, Literal):
            result = Literal(node.obj)
        else:
            result = Apply(
                node.name,
                [visit(arg) for arg in node.pos_args],
                [(key, visit(value)) for key, value in node.named_args],
                o_len=node.o_len,
                pure=node.pure,
                define_params=node.define_params,
            )
        memo[node_id] = result
        return result

    return visit(expr)


def clone_merge(expr, memo=None, merge=None):
    """Clone an expression graph, merging structurally identical pure nodes."""
    if memo is None:
        memo = {}
    if merge is None:
        merge = {}

    def visit(node):
        node_id = id(node)
        if node_id in memo:
            return memo[node_id]
        if isinstance(node, Literal):
            result = Literal(node.obj)
        else:
            args = [visit(arg) for arg in node.pos_args]
            kwargs = [(key, visit(value)) for key, value in node.named_args]
            result = Apply(
                node.name,
                args,
                kwargs,
                o_len=node.o_len,
                pure=node.pure,
                define_params=node.define_params,
            )
            if result.pure:
                key = (
                    result.name,
                    tuple(id(arg) for arg in result.pos_args),
                    tuple((name, id(value)) for name, value in result.named_args),
                    result.o_len,
                )
                result = merge.setdefault(key, result)
        memo[node_id] = result
        return result

    return visit(as_apply(expr))


def replace_input(expr, old_node, new_node):
    """Return a clone of expr with references to old_node replaced."""
    return clone(expr, replace={old_node: new_node})


def rec_eval(
    expr,
    memo=None,
    max_program_len=DEFAULT_MAX_PROGRAM_LEN,
    memo_gc=True,
    print_node_on_error=True,
):
    """Evaluate a pyll expression graph without recursive Python evaluation."""
    expr = as_apply(expr)
    if memo is None:
        memo = {}

    clients = {}
    if memo_gc:
        for node in dfs(expr):
            for child in node.inputs():
                clients[id(child)] = clients.get(id(child), 0) + 1

    stack = deque([expr])
    program_len = 0

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in memo:
            continue

        program_len += 1
        if program_len > max_program_len:
            raise RuntimeError(
                "Probably infinite loop in document: "
                "reached max_program_len=%d" % max_program_len
            )

        if isinstance(node, Literal) or node.name == "literal":
            memo[node_id] = node.obj
            continue

        if node.name == "switch":
            selector = node.pos_args[0]
            if id(selector) not in memo:
                stack.append(node)
                stack.append(selector)
                continue
            choice = memo[id(selector)]
            branch = node.pos_args[int(choice) + 1]
            if id(branch) not in memo:
                stack.append(node)
                stack.append(branch)
                continue
            memo[node_id] = memo[id(branch)]
            continue

        missing = [child for child in node.inputs() if id(child) not in memo]
        if missing:
            stack.append(node)
            stack.extend(missing)
            continue

        args = [memo[id(arg)] for arg in node.pos_args]
        kwargs = {name: memo[id(value)] for name, value in node.named_args}

        try:
            if node.name == "pos_args":
                value = tuple(args)
            elif node.name == "dict" and not node.pos_args:
                value = kwargs
            else:
                value = scope._impls[node.name](*args, **kwargs)
        except Exception:
            if print_node_on_error:
                logger.error("Error while evaluating pyll node: %s", node)
            raise

        memo[node_id] = value

        if memo_gc:
            for child in node.inputs():
                child_id = id(child)
                clients[child_id] -= 1
                if clients[child_id] <= 0 and child_id != id(expr):
                    memo.pop(child_id, None)

    return memo[id(expr)]


def expr_to_config(expr, condition, out, memo=None):
    """Extract hyperopt_param nodes and their activation conditions."""
    expr = as_apply(expr)
    if memo is None:
        memo = set()

    def visit(node, current_condition):
        node_id = id(node)
        marker = (node_id, repr(current_condition))
        if marker in memo:
            return
        memo.add(marker)

        if node.name == "hyperopt_param":
            label = node.pos_args[0].obj
            if label in out:
                old_node, old_condition = out[label]
                if old_node is not node.pos_args[1] or old_condition != current_condition:
                    raise ValueError("Duplicate hyperopt_param label", label)
            else:
                out[label] = (node.pos_args[1], current_condition)
            return

        if node.name == "switch" and len(node.pos_args) >= 2:
            selector = node.pos_args[0]
            visit(selector, current_condition)
            for index, branch in enumerate(node.pos_args[1:]):
                visit(branch, current_condition + ((selector, index),))
            return

        for child in node.inputs():
            visit(child, current_condition)

    visit(expr, tuple(condition))
    return out


def pprint(expr, ofile=None, indent=0, memo=None):
    return as_apply(expr).pprint(ofile=ofile, indent=indent, memo=memo)


@scope.define_pure
def add(a, b):
    return operator.add(a, b)


@scope.define_pure
def sub(a, b):
    return operator.sub(a, b)


@scope.define_pure
def mul(a, b):
    return operator.mul(a, b)


@scope.define_pure
def truediv(a, b):
    return operator.truediv(a, b)


@scope.define_pure
def floordiv(a, b):
    return operator.floordiv(a, b)


@scope.define_pure
def mod(a, b):
    return operator.mod(a, b)


@scope.define_pure
def pow(a, b):
    return operator.pow(a, b)


@scope.define_pure
def neg(a):
    return operator.neg(a)


@scope.define_pure
def pos(a):
    return operator.pos(a)


@scope.define_pure
def abs(a):
    return operator.abs(a)


@scope.define_pure
def invert(a):
    return operator.invert(a)


@scope.define_pure
def lt(a, b):
    return operator.lt(a, b)


@scope.define_pure
def le(a, b):
    return operator.le(a, b)


@scope.define_pure
def gt(a, b):
    return operator.gt(a, b)


@scope.define_pure
def ge(a, b):
    return operator.ge(a, b)


@scope.define_pure
def eq(a, b):
    return operator.eq(a, b)


@scope.define_pure
def ne(a, b):
    return operator.ne(a, b)


@scope.define_pure
def and_(a, b):
    return operator.and_(a, b)


@scope.define_pure
def or_(a, b):
    return operator.or_(a, b)


@scope.define_pure
def xor(a, b):
    return operator.xor(a, b)


@scope.define_pure
def getitem(a, b):
    return operator.getitem(a, b)


@scope.define
def call(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@scope.define
def switch(index, *options):
    return options[int(index)]


@scope.define_pure
def identity(x):
    return x