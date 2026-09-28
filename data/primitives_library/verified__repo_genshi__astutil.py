# -*- coding: utf-8 -*-
"""Support classes for generating code from abstract syntax trees."""

from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
                          _ast_Ellipsis, _ast_Ellipsis_value

__docformat__ = 'restructuredtext en'


def parse(source, mode):
    return compile(source, '', mode, _ast.PyCF_ONLY_AST)


class ASTCodeGenerator(object):

    binary_operators = {
        _ast.Add: '+', _ast.Sub: '-', _ast.Mult: '*', _ast.Div: '/',
        _ast.FloorDiv: '//', _ast.Mod: '%', _ast.Pow: '**',
        _ast.LShift: '<<', _ast.RShift: '>>', _ast.BitOr: '|',
        _ast.BitXor: '^', _ast.BitAnd: '&', _ast.MatMult: '@'
    }

    unary_operators = {
        _ast.Invert: '~', _ast.Not: 'not ', _ast.UAdd: '+', _ast.USub: '-'
    }

    boolean_operators = {
        _ast.And: 'and', _ast.Or: 'or'
    }

    comparison_operators = {
        _ast.Eq: '==', _ast.NotEq: '!=', _ast.Lt: '<', _ast.LtE: '<=',
        _ast.Gt: '>', _ast.GtE: '>=', _ast.Is: 'is', _ast.IsNot: 'is not',
        _ast.In: 'in', _ast.NotIn: 'not in'
    }

    def __init__(self, tree):
        self.lines_info = []
        self.line_info = None
        self.code = ''
        self.line = None
        self.last = None
        self.indent = 0
        self.blame_stack = []
        self.visit(tree)
        if self.line.strip():
            self.code += self.line + '\n'
            self.lines_info.append(self.line_info)
        self.line = None
        self.line_info = None

    def _change_indent(self, delta):
        self.indent += delta

    def _new_line(self):
        if self.line is not None:
            self.code += self.line + '\n'
            self.lines_info.append(self.line_info)
        self.line = ' ' * 4 * self.indent
        if not self.blame_stack:
            self.line_info = []
            self.last = None
        else:
            self.line_info = [(0, self.blame_stack[-1])]
            self.last = self.blame_stack[-1]

    def _write(self, s):
        if not s:
            return
        if not self.blame_stack:
            if self.last is not None:
                self.last = None
                self.line_info.append((len(self.line), self.last))
        elif self.last != self.blame_stack[-1]:
            self.last = self.blame_stack[-1]
            self.line_info.append((len(self.line), self.last))
        self.line += s

    def visit(self, node):
        if node is None:
            return None
        if type(node) is tuple:
            return tuple(self.visit(item) for item in node)
        try:
            self.blame_stack.append((node.lineno, node.col_offset))
            has_info = True
        except AttributeError:
            has_info = False
        if isinstance(node, (bool, bytes, float, int, str)):
            node = _ast_Constant(node)
        method = getattr(self, 'visit_' + node.__class__.__name__, None)
        if method is None:
            raise Exception('Unhandled node type %r' % type(node))
        result = method(node)
        if has_info:
            self.blame_stack.pop()
        return result

    def visit_Module(self, node):
        for statement in node.body:
            self.visit(statement)

    visit_Interactive = visit_Module
    visit_Suite = visit_Module

    def visit_Expression(self, node):
        self._new_line()
        return self.visit(node.body)

    def visit_arguments(self, node):
        first = [True]

        def comma():
            if first[0]:
                first[0] = False
            else:
                self._write(', ')

        def arguments(args, defaults):
            bare = len(args) - len(defaults)
            for index, arg in enumerate(args):
                comma()
                self.visit(arg)
                default_index = index - bare
                if default_index >= 0 and defaults[default_index] is not None:
                    self._write('=')
                    self.visit(defaults[default_index])

        posonly = getattr(node, 'posonlyargs', ())
        if posonly:
            arguments(posonly, ())
            comma()
            self._write('/')

        arguments(node.args, node.defaults)
        vararg = getattr(node, 'vararg', None)
        kwonly = getattr(node, 'kwonlyargs', None)
        if vararg:
            comma()
            self._write('*')
            if isstring(vararg):
                self._write(vararg)
            else:
                self.visit(vararg)
        elif kwonly:
            comma()
            self._write('*')
        if kwonly:
            arguments(kwonly, node.kw_defaults)
        kwarg = getattr(node, 'kwarg', None)
        if kwarg:
            comma()
            self._write('**')
            if isstring(kwarg):
                self._write(kwarg)
            else:
                self.visit(kwarg)

    if not IS_PYTHON2:
        def visit_arg(self, node):
            self._write(node.arg)

    def visit_Starred(self, node):
        self._write('*')
        self.visit(node.value)

    def _visit_decorators(self, node):
        decorators = getattr(node, 'decorator_list',
                             getattr(node, 'decorators', ()))
        for decorator in decorators:
            self._new_line()
            self._write('@')
            self.visit(decorator)

    def visit_FunctionDef(self, node):
        self._visit_decorators(node)
        self._new_line()
        self._write('def ' + node.name + '(')
        self.visit(node.args)
        self._write(')')
        if getattr(node, 'returns', None) is not None:
            self._write(' -> ')
            self.visit(node.returns)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_AsyncFunctionDef(self, node):
        self._visit_decorators(node)
        self._new_line()
        self._write('async def ' + node.name + '(')
        self.visit(node.args)
        self._write(')')
        if getattr(node, 'returns', None) is not None:
            self._write(' -> ')
            self.visit(node.returns)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_ClassDef(self, node):
        self._visit_decorators(node)
        self._new_line()
        self._write('class ' + node.name)
        bases = list(node.bases)
        keywords = getattr(node, 'keywords', ())
        if bases or keywords:
            self._write('(')
            first = True
            for base in bases:
                if not first:
                    self._write(', ')
                self.visit(base)
                first = False
            for keyword in keywords:
                if not first:
                    self._write(', ')
                self.visit(keyword)
                first = False
            self._write(')')
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_Return(self, node):
        self._new_line()
        self._write('return')
        if getattr(node, 'value', None):
            self._write(' ')
            self.visit(node.value)

    def visit_Delete(self, node):
        self._new_line()
        self._write('del ')
        self._visit_sequence(node.targets)

    def visit_Assign(self, node):
        self._new_line()
        for target in node.targets:
            self.visit(target)
            self._write(' = ')
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self._new_line()
        self.visit(node.target)
        self._write(': ')
        self.visit(node.annotation)
        if node.value is not None:
            self._write(' = ')
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self._new_line()
        self.visit(node.target)
        self._write(' ' + self.binary_operators[node.op.__class__] + '= ')
        self.visit(node.value)

    def visit_Print(self, node):
        self._new_line()
        self._write('print')
        if getattr(node, 'dest', None):
            self._write(' >> ')
            self.visit(node.dest)
            if getattr(node, 'values', None):
                self._write(', ')
        else:
            self._write(' ')
        if getattr(node, 'values', None):
            self._visit_sequence(node.values)
        if not node.nl:
            self._write(',')

    def visit_For(self, node):
        self._new_line()
        self._write('for ')
        self.visit(node.target)
        self._write(' in ')
        self.visit(node.iter)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'orelse', None):
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_AsyncFor(self, node):
        self._new_line()
        self._write('async for ')
        self.visit(node.target)
        self._write(' in ')
        self.visit(node.iter)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if node.orelse:
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_While(self, node):
        self._new_line()
        self._write('while ')
        self.visit(node.test)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'orelse', None):
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_If(self, node):
        self._new_line()
        self._write('if ')
        self.visit(node.test)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], _ast.If):
                self._write('else:')
                self._change_indent(1)
                self.visit(node.orelse[0])
                self._change_indent(-1)
            else:
                self._new_line()
                self._write('else:')
                self._change_indent(1)
                for statement in node.orelse:
                    self.visit(statement)
                self._change_indent(-1)

    def visit_With(self, node):
        self._new_line()
        self._write('with ')
        if hasattr(node, 'items'):
            for index, item in enumerate(node.items):
                if index:
                    self._write(', ')
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    self._write(' as ')
                    self.visit(item.optional_vars)
        else:
            self.visit(node.context_expr)
            if node.optional_vars is not None:
                self._write(' as ')
                self.visit(node.optional_vars)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_AsyncWith(self, node):
        self._new_line()
        self._write('async with ')
        for index, item in enumerate(node.items):
            if index:
                self._write(', ')
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._write(' as ')
                self.visit(item.optional_vars)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_Raise(self, node):
        self._new_line()
        self._write('raise')
        if hasattr(node, 'exc'):
            if node.exc is not None:
                self._write(' ')
                self.visit(node.exc)
            if node.cause is not None:
                self._write(' from ')
                self.visit(node.cause)
        elif getattr(node, 'type', None):
            self._write(' ')
            self.visit(node.type)
            if getattr(node, 'inst', None):
                self._write(', ')
                self.visit(node.inst)
            if getattr(node, 'tback', None):
                self._write(', ')
                self.visit(node.tback)

    def visit_Try(self, node):
        self._new_line()
        self._write('try:')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        for handler in node.handlers:
            self.visit(handler)
        if node.orelse:
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)
        if node.finalbody:
            self._new_line()
            self._write('finally:')
            self._change_indent(1)
            for statement in node.finalbody:
                self.visit(statement)
            self._change_indent(-1)

    visit_TryFinally = visit_Try
    visit_TryExcept = visit_Try

    def visit_ExceptHandler(self, node):
        self._new_line()
        self._write('except')
        if node.type is not None:
            self._write(' ')
            self.visit(node.type)
            if node.name:
                self._write(' as ' + (node.name if isstring(node.name)
                                      else node.name.id))
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_Assert(self, node):
        self._new_line()
        self._write('assert ')
        self.visit(node.test)
        if node.msg is not None:
            self._write(', ')
            self.visit(node.msg)

    def visit_Import(self, node):
        self._new_line()
        self._write('import ')
        self._visit_sequence(node.names)

    def visit_ImportFrom(self, node):
        self._new_line()
        self._write('from ')
        self._write('.' * getattr(node, 'level', 0))
        if node.module:
            self._write(node.module)
        self._write(' import ')
        self._visit_sequence(node.names)

    def visit_Global(self, node):
        self._new_line()
        self._write('global ' + ', '.join(node.names))

    def visit_Nonlocal(self, node):
        self._new_line()
        self._write('nonlocal ' + ', '.join(node.names))

    def visit_Expr(self, node):
        self._new_line()
        self.visit(node.value)

    def visit_Pass(self, node):
        self._new_line()
        self._write('pass')

    def visit_Break(self, node):
        self._new_line()
        self._write('break')

    def visit_Continue(self, node):
        self._new_line()
        self._write('continue')

    def visit_alias(self, node):
        self._write(node.name)
        if node.asname:
            self._write(' as ' + node.asname)

    def visit_BoolOp(self, node):
        operator = ' ' + self.boolean_operators[node.op.__class__] + ' '
        self.visit(node.values[0])
        for value in node.values[1:]:
            self._write(operator)
            self.visit(value)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self._write(' ' + self.binary_operators[node.op.__class__] + ' ')
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self._write(self.unary_operators[node.op.__class__])
        self.visit(node.operand)

    def visit_Lambda(self, node):
        self._write('lambda ')
        self.visit(node.args)
        self._write(': ')
        self.visit(node.body)

    def visit_IfExp(self, node):
        self.visit(node.body)
        self._write(' if ')
        self.visit(node.test)
        self._write(' else ')
        self.visit(node.orelse)

    def visit_Dict(self, node):
        self._write('{')
        first = True
        for key, value in zip(node.keys, node.values):
            if not first:
                self._write(', ')
            if key is None:
                self._write('**')
                self.visit(value)
            else:
                self.visit(key)
                self._write(': ')
                self.visit(value)
            first = False
        self._write('}')

    def visit_Set(self, node):
        self._write('{')
        self._visit_sequence(node.elts)
        self._write('}')

    def visit_ListComp(self, node):
        self._write('[')
        self.visit(node.elt)
        self._visit_generators(node.generators)
        self._write(']')

    def visit_SetComp(self, node):
        self._write('{')
        self.visit(node.elt)
        self._visit_generators(node.generators)
        self._write('}')

    def visit_GeneratorExp(self, node):
        self._write('(')
        self.visit(node.elt)
        self._visit_generators(node.generators)
        self._write(')')

    def visit_DictComp(self, node):
        self._write('{')
        self.visit(node.key)
        self._write(': ')
        self.visit(node.value)
        self._visit_generators(node.generators)
        self._write('}')

    def _visit_generators(self, generators):
        for generator in generators:
            self._write(' ')
            self.visit(generator)

    def visit_comprehension(self, node):
        if getattr(node, 'is_async', False):
            self._write('async ')
        self._write('for ')
        self.visit(node.target)
        self._write(' in ')
        self.visit(node.iter)
        for test in node.ifs:
            self._write(' if ')
            self.visit(test)

    def visit_Yield(self, node):
        self._write('yield')
        if node.value is not None:
            self._write(' ')
            self.visit(node.value)

    def visit_YieldFrom(self, node):
        self._write('yield from ')
        self.visit(node.value)

    def visit_Compare(self, node):
        self.visit(node.left)
        for operator, comparator in zip(node.ops, node.comparators):
            self._write(' ' + self.comparison_operators[operator.__class__] + ' ')
            self.visit(comparator)

    def visit_Call(self, node):
        self.visit(node.func)
        self._write('(')
        first = True

        def separator():
            nonlocal first
            if first:
                first = False
            else:
                self._write(', ')

        for argument in node.args:
            separator()
            self.visit(argument)
        for keyword in node.keywords:
            separator()
            self.visit(keyword)
        for argument in getattr(node, 'starargs', ()) or ():
            separator()
            self._write('*')
            self.visit(argument)
        kwargs = getattr(node, 'kwargs', None)
        if kwargs is not None:
            separator()
            self._write('**')
            self.visit(kwargs)
        self._write(')')

    def visit_keyword(self, node):
        if node.arg is None:
            self._write('**')
        else:
            self._write(node.arg + '=')
        self.visit(node.value)

    def visit_FormattedValue(self, node):
        self._write('{')
        self.visit(node.value)
        if node.conversion != -1:
            self._write('!' + chr(node.conversion))
        if node.format_spec is not None:
            self._write(':')
            self.visit(node.format_spec)
        self._write('}')

    def visit_JoinedStr(self, node):
        self._write('f')
        self._write(repr(''.join(
            value.value if isinstance(value, _ast.Constant) and
            isinstance(value.value, str) else ''
            for value in node.values
        )))
        # Replace the simple representation with a fully rendered form.
        rendered = []
        for value in node.values:
            if isinstance(value, _ast.Constant) and isinstance(value.value, str):
                rendered.append(value.value.replace('{', '{{').replace('}', '}}'))
            else:
                class Writer(object):
                    def __init__(self):
                        self.value = ''
                    def _write(self, text):
                        self.value += text
                old_write = self._write
                collector = Writer()
                self._write = collector._write
                try:
                    self.visit(value)
                finally:
                    self._write = old_write
                rendered.append(collector.value)
        text = 'f' + repr(''.join(rendered))
        self.line = self.line[:-len('f' + repr(''.join(
            value.value if isinstance(value, _ast.Constant) and
            isinstance(value.value, str) else ''
            for value in node.values
        )))] + text

    def visit_Attribute(self, node):
        self.visit(node.value)
        self._write('.' + node.attr)

    def visit_Subscript(self, node):
        self.visit(node.value)
        self._write('[')
        self.visit(node.slice)
        self._write(']')

    def visit_Name(self, node):
        self._write(node.id)

    def visit_List(self, node):
        self._write('[')
        self._visit_sequence(node.elts)
        self._write(']')

    def visit_Tuple(self, node):
        self._write('(')
        self._visit_sequence(node.elts)
        if len(node.elts) == 1:
            self._write(',')
        self._write(')')

    def visit_Slice(self, node):
        if node.lower is not None:
            self.visit(node.lower)
        self._write(':')
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self._write(':')
            self.visit(node.step)

    def visit_Index(self, node):
        self.visit(node.value)

    def visit_ExtSlice(self, node):
        self._visit_sequence(node.dims)

    def visit_NameConstant(self, node):
        self._write(repr(node.value))

    def visit_Num(self, node):
        self._write(repr(node.n))

    def visit_Str(self, node):
        self._write(repr(node.s))

    def visit_Bytes(self, node):
        self._write(repr(node.s))

    def visit_Ellipsis(self, node):
        self._write('...')

    def visit_Constant(self, node):
        value = node.value
        if _ast_Ellipsis is not None and value is _ast_Ellipsis_value:
            self._write('...')
        else:
            self._write(repr(value))

    def visit_Await(self, node):
        self._write('await ')
        self.visit(node.value)

    def visit_Match(self, node):
        self._new_line()
        self._write('match ')
        self.visit(node.subject)
        self._write(':')
        self._change_indent(1)
        for case in node.cases:
            self.visit(case)
        self._change_indent(-1)

    def visit_match_case(self, node):
        self._new_line()
        self._write('case ')
        self.visit(node.pattern)
        if node.guard is not None:
            self._write(' if ')
            self.visit(node.guard)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_MatchValue(self, node):
        self.visit(node.value)

    def visit_MatchSingleton(self, node):
        self._write(repr(node.value))

    def visit_MatchSequence(self, node):
        self._write('[')
        self._visit_sequence(node.patterns)
        self._write(']')

    def visit_MatchStar(self, node):
        self._write('*')
        if node.name:
            self._write(node.name)

    def visit_MatchAs(self, node):
        if node.pattern is not None:
            self.visit(node.pattern)
            if node.name:
                self._write(' as ')
        if node.name:
            self._write(node.name)
        elif node.pattern is None:
            self._write('_')

    def visit_MatchOr(self, node):
        self.visit(node.patterns[0])
        for pattern in node.patterns[1:]:
            self._write(' | ')
            self.visit(pattern)

    def visit_MatchMapping(self, node):
        self._write('{')
        first = True
        for key, pattern in zip(node.keys, node.patterns):
            if not first:
                self._write(', ')
            self.visit(key)
            self._write(': ')
            self.visit(pattern)
            first = False
        if node.rest:
            if not first:
                self._write(', ')
            self._write('**' + node.rest)
        self._write('}')

    def visit_MatchClass(self, node):
        self.visit(node.cls)
        self._write('(')
        first = True
        for pattern in node.patterns:
            if not first:
                self._write(', ')
            self.visit(pattern)
            first = False
        for name, pattern in zip(node.kwd_attrs, node.kwd_patterns):
            if not first:
                self._write(', ')
            self._write(name + '=')
            self.visit(pattern)
            first = False
        self._write(')')

    def _visit_sequence(self, values):
        for index, value in enumerate(values):
            if index:
                self._write(', ')
            self.visit(value)