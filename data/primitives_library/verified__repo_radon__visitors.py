import ast
import collections
import operator

GET_COMPLEXITY = operator.attrgetter('complexity')
GET_REAL_COMPLEXITY = operator.attrgetter('real_complexity')
NAMES_GETTER = operator.attrgetter('name', 'asname')
GET_ENDLINE = operator.attrgetter('endline')

BaseFunc = collections.namedtuple(
    'Function',
    (
        'name',
        'lineno',
        'col_offset',
        'endline',
        'is_method',
        'classname',
        'closures',
        'complexity',
    ),
)

BaseClass = collections.namedtuple(
    'Class',
    (
        'name',
        'lineno',
        'col_offset',
        'endline',
        'methods',
        'inner_classes',
        'real_complexity',
    ),
)


def code2ast(source):
    return ast.parse(source)


class Function(BaseFunc):
    @property
    def letter(self):
        return 'M' if self.is_method else 'F'

    @property
    def fullname(self):
        if self.classname is None:
            return self.name
        return '{0}.{1}'.format(self.classname, self.name)

    def __str__(self):
        return '{0} {1}:{2}->{3} {4} - {5}'.format(
            self.letter,
            self.lineno,
            self.col_offset,
            self.endline,
            self.fullname,
            self.complexity,
        )


class Class(BaseClass):
    letter = 'C'

    @property
    def fullname(self):
        return self.name

    @property
    def complexity(self):
        if not self.methods:
            return self.real_complexity
        methods = len(self.methods)
        return int(self.real_complexity / float(methods)) + (methods > 1)

    def __str__(self):
        return '{0} {1}:{2}->{3} {4} - {5}'.format(
            self.letter,
            self.lineno,
            self.col_offset,
            self.endline,
            self.name,
            self.complexity,
        )


class CodeVisitor(ast.NodeVisitor):
    @staticmethod
    def get_name(obj):
        return obj.__class__.__name__

    @classmethod
    def from_code(cls, code, **kwargs):
        return cls.from_ast(code2ast(code), **kwargs)

    @classmethod
    def from_ast(cls, ast_node, **kwargs):
        visitor = cls(**kwargs)
        visitor.visit(ast_node)
        return visitor


class ComplexityVisitor(CodeVisitor):
    def __init__(
        self, to_method=False, classname=None, off=True, no_assert=False
    ):
        self.off = off
        self.complexity = 1 if off else 0
        self.functions = []
        self.classes = []
        self.to_method = to_method
        self.classname = classname
        self.no_assert = no_assert
        self._max_line = float('-inf')

    @property
    def functions_complexity(self):
        return sum(map(GET_COMPLEXITY, self.functions)) - len(self.functions)

    @property
    def classes_complexity(self):
        return (
            sum(map(GET_REAL_COMPLEXITY, self.classes)) - len(self.classes)
        )

    @property
    def total_complexity(self):
        return (
            self.complexity
            + self.functions_complexity
            + self.classes_complexity
            + (not self.off)
        )

    @property
    def blocks(self):
        result = []
        result.extend(self.functions)
        for cls in self.classes:
            result.append(cls)
            result.extend(cls.methods)
        return result

    @property
    def max_line(self):
        return self._max_line

    @max_line.setter
    def max_line(self, value):
        if value > self._max_line:
            self._max_line = value

    def generic_visit(self, node):
        name = self.get_name(node)

        if hasattr(node, 'lineno'):
            self.max_line = node.lineno

        if name in ('Try', 'TryExcept'):
            self.complexity += len(node.handlers) + bool(node.orelse)
        elif name == 'BoolOp':
            self.complexity += len(node.values) - 1
        elif name in ('If', 'IfExp'):
            self.complexity += 1
        elif name == 'Match':
            contain_underscore = any(
                case
                for case in node.cases
                if getattr(case.pattern, 'pattern', False) is None
            )
            self.complexity += max(0, len(node.cases) - contain_underscore)
        elif name in ('For', 'While', 'AsyncFor'):
            self.complexity += bool(node.orelse) + 1
        elif name == 'comprehension':
            self.complexity += len(node.ifs) + 1

        super().generic_visit(node)

    def visit_Assert(self, node):
        self.complexity += not self.no_assert

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        visitor = self.__class__(
            to_method=True,
            classname=self.classname,
            off=False,
            no_assert=self.no_assert,
        )
        for child in node.body:
            visitor.visit(child)

        function = Function(
            node.name,
            node.lineno,
            node.col_offset,
            visitor.max_line,
            self.to_method,
            self.classname,
            visitor.functions,
            visitor.complexity + 1,
        )
        self.functions.append(function)

    def visit_ClassDef(self, node):
        visitor = self.__class__(
            to_method=True,
            classname=node.name,
            off=False,
            no_assert=self.no_assert,
        )
        for child in node.body:
            visitor.visit(child)

        methods = visitor.functions
        inner_classes = visitor.classes
        real_complexity = (
            visitor.complexity
            + sum(map(GET_COMPLEXITY, methods))
            + sum(map(GET_REAL_COMPLEXITY, inner_classes))
        )

        cls = Class(
            node.name,
            node.lineno,
            node.col_offset,
            visitor.max_line,
            methods,
            inner_classes,
            real_complexity,
        )
        self.classes.append(cls)


class HalsteadVisitor(CodeVisitor):
    def __init__(self):
        self.operators = 0
        self.operands = 0
        self._distinct_operators = set()
        self._distinct_operands = set()
        self.function_visitors = []

    @property
    def distinct_operators(self):
        return len(self._distinct_operators)

    @property
    def distinct_operands(self):
        return len(self._distinct_operands)

    def _add_operator(self, operator):
        self.operators += 1
        self._distinct_operators.add(self.get_name(operator))

    def _add_operand(self, operand):
        self.operands += 1
        if isinstance(operand, ast.Name):
            self._distinct_operands.add(operand.id)
        else:
            self._distinct_operands.add(operand)

    def _add_operands(self, *operands):
        for operand in operands:
            self._add_operand(operand)

    def visit_BinOp(self, node):
        self._add_operator(node.op)
        self._add_operands(node.left, node.right)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self._add_operator(node.op)
        self._add_operands(*node.values)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._add_operator(node.op)
        self._add_operands(node.target, node.value)
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self._add_operator(op)
        self._add_operands(node.left, *node.comparators)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self._add_operator(node.op)
        self._add_operands(node.operand)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        visitor = self.__class__()
        visitor.name = node.name
        for child in node.body:
            visitor.visit(child)
        self.function_visitors.append(visitor)