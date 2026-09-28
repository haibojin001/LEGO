from __future__ import annotations

from collections import deque
from math import ceil, floor
import re

from genshi.compat import text_type
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import (START, END, TEXT, START_NS, END_NS, COMMENT, PI,
                         START_CDATA, END_CDATA)

__all__ = ['Path', 'PathSyntaxError']
__docformat__ = 'restructuredtext en'


class PathSyntaxError(Exception):
    def __init__(self, message, filename=None, lineno=-1, offset=-1, text=None):
        self.msg = message
        self.filename = filename
        self.lineno = lineno
        self.offset = offset
        self.text = text
        Exception.__init__(self, message)

    def __str__(self):
        if self.filename:
            return '%s (%s, line %s)' % (self.msg, self.filename, self.lineno)
        return self.msg


class Axis(object):
    ATTRIBUTE = 'attribute'
    CHILD = 'child'
    DESCENDANT = 'descendant'
    DESCENDANT_OR_SELF = 'descendant-or-self'
    SELF = 'self'

    @classmethod
    def forname(cls, name):
        return getattr(cls, name.upper().replace('-', '_'), None)


ATTRIBUTE = Axis.ATTRIBUTE
CHILD = Axis.CHILD
DESCENDANT = Axis.DESCENDANT
DESCENDANT_OR_SELF = Axis.DESCENDANT_OR_SELF
SELF = Axis.SELF


def _qname_parts(value):
    value = text_type(value)
    if value.startswith('{'):
        end = value.find('}')
        if end >= 0:
            return value[1:end], value[end + 1:]
    return None, value


def _attrs_dict(attrs):
    result = {}
    if attrs is None:
        return result
    try:
        iterator = attrs
        for name, value in iterator:
            result[text_type(name)] = value
    except (TypeError, ValueError):
        try:
            result.update(attrs)
        except Exception:
            pass
    return result


class _Node(object):
    __slots__ = ('kind', 'name', 'attrs', 'data', 'parent', 'children',
                 'start', 'end', 'event', 'pos')

    def __init__(self, kind, name=None, attrs=None, data=None, parent=None,
                 event=None, pos=None):
        self.kind = kind
        self.name = name
        self.attrs = attrs or {}
        self.data = data
        self.parent = parent
        self.children = []
        self.start = None
        self.end = None
        self.event = event
        self.pos = pos


class _AttributeNode(_Node):
    __slots__ = ()

    def __init__(self, name, value, parent):
        _Node.__init__(self, 'attribute', name, data=value, parent=parent,
                       pos=parent.pos)


class _NodeTest(object):
    def __call__(self, kind, data, pos, namespaces, variables):
        return False


class LocalNameTest(_NodeTest):
    def __init__(self, name):
        self.name = name

    def __call__(self, kind, data, pos, namespaces, variables):
        if kind is not START:
            return False
        name = data[0] if isinstance(data, tuple) else data
        return _qname_parts(name)[1] == self.name


class QualifiedNameTest(_NodeTest):
    def __init__(self, name):
        self.name = name

    def __call__(self, kind, data, pos, namespaces, variables):
        if kind is not START:
            return False
        name = data[0] if isinstance(data, tuple) else data
        return text_type(name) == self.name


class PrincipalTypeTest(_NodeTest):
    def __call__(self, kind, data, pos, namespaces, variables):
        return kind is START


class TextNodeTest(_NodeTest):
    def __call__(self, kind, data, pos, namespaces, variables):
        return kind is TEXT


class CommentNodeTest(_NodeTest):
    def __call__(self, kind, data, pos, namespaces, variables):
        return kind is COMMENT


class ProcessingInstructionNodeTest(_NodeTest):
    def __init__(self, target=None):
        self.target = target

    def __call__(self, kind, data, pos, namespaces, variables):
        if kind is not PI:
            return False
        if self.target is None:
            return True
        try:
            return data[0] == self.target
        except (TypeError, IndexError):
            return False


class NodeTest(_NodeTest):
    def __call__(self, kind, data, pos, namespaces, variables):
        return kind in (START, TEXT, COMMENT, PI)


class GenericStrategy(object):
    @classmethod
    def supports(cls, path):
        return True

    def __init__(self, path):
        self.path = path

    def test(self, ignore_context=False):
        matcher = _IncrementalMatcher(self.path, ignore_context)
        return matcher


class SimplePathStrategy(GenericStrategy):
    @classmethod
    def supports(cls, path):
        return True


class _IncrementalMatcher(object):
    """A small compatibility matcher used by code that calls Path.test()."""

    def __init__(self, path, ignore_context=False):
        self.path = path
        self.ignore_context = ignore_context
        self.stack = []
        self.depth = 0

    def __call__(self, event, namespaces=None, variables=None,
                 updateonly=False):
        kind = event[0]
        if kind is END:
            self.depth -= 1
            return None
        if kind is not START:
            return None
        self.depth += 1
        try:
            test = self.path[-1][1]
            return bool(test(kind, event[1], event[2], namespaces or {},
                             variables or {}))
        except Exception:
            return None


def _build_tree(events):
    root = _Node('document')
    stack = [root]
    nodes = []
    for index, event in enumerate(events):
        kind = event[0]
        data = event[1] if len(event) > 1 else None
        pos = event[2] if len(event) > 2 else None
        if kind is START:
            try:
                name, attrs = data
            except (TypeError, ValueError):
                name, attrs = data, ()
            node = _Node('element', name, _attrs_dict(attrs), parent=stack[-1],
                         event=index, pos=pos)
            node.start = index
            stack[-1].children.append(node)
            stack.append(node)
            nodes.append(node)
        elif kind is END:
            if len(stack) > 1:
                node = stack.pop()
                node.end = index
        elif kind is TEXT:
            node = _Node('text', data=data, parent=stack[-1], event=index,
                         pos=pos)
            node.start = node.end = index
            stack[-1].children.append(node)
            nodes.append(node)
        elif kind is COMMENT:
            node = _Node('comment', data=data, parent=stack[-1], event=index,
                         pos=pos)
            node.start = node.end = index
            stack[-1].children.append(node)
            nodes.append(node)
        elif kind is PI:
            node = _Node('pi', data=data, parent=stack[-1], event=index,
                         pos=pos)
            node.start = node.end = index
            stack[-1].children.append(node)
            nodes.append(node)
    root.start = 0
    root.end = len(events) - 1
    return root


def _node_string(node):
    if isinstance(node, _AttributeNode):
        return text_type(node.data)
    if node.kind in ('text', 'comment'):
        return text_type(node.data or '')
    if node.kind == 'pi':
        try:
            return text_type(node.data[1])
        except Exception:
            return text_type(node.data or '')
    if node.kind == 'element' or node.kind == 'document':
        return ''.join(_node_string(child) for child in node.children
                       if child.kind in ('text', 'element'))
    return ''


def _boolean(value):
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, float):
        return value != 0 and value == value
    return bool(value)


def _number(value):
    if isinstance(value, list):
        value = _node_string(value[0]) if value else ''
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _string(value):
    if isinstance(value, list):
        return _node_string(value[0]) if value else ''
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return text_type(value)


def _compare(left, right, op):
    lefts = left if isinstance(left, list) else [left]
    rights = right if isinstance(right, list) else [right]
    for a in lefts:
        for b in rights:
            av = _node_string(a) if isinstance(a, _Node) else a
            bv = _node_string(b) if isinstance(b, _Node) else b
            if op in ('=', '!='):
                equal = _string(av) == _string(bv)
                if (op == '=' and equal) or (op == '!=' and not equal):
                    return True
            else:
                an, bn = _number(av), _number(bv)
                if op == '<' and an < bn:
                    return True
                if op == '<=' and an <= bn:
                    return True
                if op == '>' and an > bn:
                    return True
                if op == '>=' and an >= bn:
                    return True
    return False


_token_re = re.compile(
    r'\s*(?:(?P<string>"[^"]*"|\'[^\']*\')|(?P<number>(?:\d+\.\d*|\.\d+|\d+))|'
    r'(?P<op>//|::|!=|<=|>=|\.\.|[()\[\],/@$=<>+\-*.])|'
    r'(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*|\*))'
)


class _ExpressionParser(object):
    def __init__(self, source, error):
        self.source = source
        self.error = error
        self.tokens = []
        at = 0
        while at < len(source):
            match = _token_re.match(source, at)
            if not match:
                raise error('Invalid predicate expression')
            at = match.end()
            value = match.group('string') or match.group('number') or \
                    match.group('op') or match.group('name')
            self.tokens.append(value)
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, value=None):
        token = self.peek()
        if token is None or (value is not None and token != value):
            raise self.error('Expected %s' % (value or 'expression'))
        self.index += 1
        return token

    def parse(self):
        value = self.or_expr()
        if self.peek() is not None:
            raise self.error('Unexpected token %s' % self.peek())
        return value

    def or_expr(self):
        value = self.and_expr()
        while self.peek() == 'or':
            self.take()
            value = ('or', value, self.and_expr())
        return value

    def and_expr(self):
        value = self.compare_expr()
        while self.peek() == 'and':
            self.take()
            value = ('and', value, self.compare_expr())
        return value

    def compare_expr(self):
        value = self.add_expr()
        while self.peek() in ('=', '!=', '<', '<=', '>', '>='):
            op = self.take()
            value = ('compare', op, value, self.add_expr())
        return value

    def add_expr(self):
        value = self.mul_expr()
        while self.peek() in ('+', '-'):
            op = self.take()
            value = ('math', op, value, self.mul_expr())
        return value

    def mul_expr(self):
        value = self.unary_expr()
        while self.peek() in ('*', 'div', 'mod'):
            op = self.take()
            value = ('math', op, value, self.unary_expr())
        return value

    def unary_expr(self):
        if self.peek() == '-':
            self.take()
            return ('neg', self.unary_expr())
        return self.primary()

    def primary(self):
        token = self.take()
        if token[0:1] in ('"', "'"):
            return ('literal', token[1:-1])
        try:
            return ('literal', float(token))
        except ValueError:
            pass
        if token == '(':
            value = self.or_expr()
            self.take(')')
            return value
        if token == '$':
            return ('variable', self.take())
        if token == '@':
            return ('path', [('attribute', self.take())])
        if token == '.':
            return ('self',)
        if token == 'true' and self.peek() == '(':
            self.take('(')
            self.take(')')
            return ('literal', True)
        if token == 'false' and self.peek() == '(':
            self.take('(')
            self.take(')')
            return ('literal', False)
        if self.peek() == '(':
            self.take('(')
            args = []
            if self.peek() != ')':
                while True:
                    args.append(self.or_expr())
                    if self.peek() != ',':
                        break
                    self.take(',')
            self.take(')')
            return ('function', token, args)
        steps = [('child', token)]
        while self.peek() == '/':
            self.take('/')
            if self.peek() == '@':
                self.take('@')
                steps.append(('attribute', self.take()))
            else:
                steps.append(('child', self.take()))
        return ('path', steps)


def _eval_expr(expr, node, position, size, namespaces, variables):
    kind = expr[0]
    if kind == 'literal':
        return expr[1]
    if kind == 'variable':
        return variables.get(expr[1])
    if kind == 'self':
        return [node]
    if kind == 'path':
        current = [node]
        for axis, name in expr[1]:
            following = []
            for item in current:
                if axis == 'attribute':
                    for attrname, value in item.attrs.items():
                        attr = _AttributeNode(attrname, value, item)
                        if _matches_name(attr, name, namespaces):
                            following.append(attr)
                else:
                    for child in item.children:
                        if _matches_name(child, name, namespaces):
                            following.append(child)
            current = following
        return current
    if kind == 'or':
        return _boolean(_eval_expr(expr[1], node, position, size, namespaces,
                                   variables)) or _boolean(_eval_expr(
                                       expr[2], node, position, size,
                                       namespaces, variables))
    if kind == 'and':
        return _boolean(_eval_expr(expr[1], node, position, size, namespaces,
                                   variables)) and _boolean(_eval_expr(
                                       expr[2], node, position, size,
                                       namespaces, variables))
    if kind == 'compare':
        return _compare(_eval_expr(expr[2], node, position, size, namespaces,
                                   variables),
                        _eval_expr(expr[3], node, position, size, namespaces,
                                   variables), expr[1])
    if kind == 'neg':
        return -_number(_eval_expr(expr[1], node, position, size, namespaces,
                                   variables))
    if kind == 'math':
        left = _number(_eval_expr(expr[2], node, position, size, namespaces,
                                  variables))
        right = _number(_eval_expr(expr[3], node, position, size, namespaces,
                                   variables))
        try:
            if expr[1] == '+':
                return left + right
            if expr[1] == '-':
                return left - right
            if expr[1] == '*':
                return left * right
            if expr[1] == 'div':
                return left / right
            return left % right
        except ZeroDivisionError:
            return float('nan')
    if kind == 'function':
        name = expr[1]
        args = [_eval_expr(arg, node, position, size, namespaces, variables)
                for arg in expr[2]]
        if name == 'position':
            return float(position)
        if name == 'last':
            return float(size)
        if name == 'not':
            return not _boolean(args[0])
        if name == 'boolean':
            return _boolean(args[0])
        if name == 'string':
            return _string(args[0] if args else [node])
        if name == 'number':
            return _number(args[0] if args else [node])
        if name == 'count':
            return float(len(args[0])) if isinstance(args[0], list) else 0.0
        if name == 'contains':
            return _string(args[1]) in _string(args[0])
        if name == 'starts-with':
            return _string(args[0]).startswith(_string(args[1]))
        if name == 'ends-with':
            return _string(args[0]).endswith(_string(args[1]))
        if name == 'concat':
            return ''.join(_string(arg) for arg in args)
        if name == 'substring-before':
            text, part = _string(args[0]), _string(args[1])
            return text.split(part, 1)[0] if part in text else ''
        if name == 'substring-after':
            text, part = _string(args[0]), _string(args[1])
            return text.split(part, 1)[1] if part in text else ''
        if name == 'substring':
            text = _string(args[0])
            start = int(round(_number(args[1]))) - 1
            if len(args) > 2:
                return text[start:start + int(round(_number(args[2])))]
            return text[start:]
        if name == 'string-length':
            return float(len(_string(args[0] if args else [node])))
        if name == 'normalize-space':
            return ' '.join(_string(args[0] if args else [node]).split())
        if name == 'translate':
            text, source, destination = map(_string, args[:3])
            table = {}
            for index, char in enumerate(source):
                table[char] = destination[index] if index < len(destination) else ''
            return ''.join(table.get(char, char) for char in text)
        if name == 'floor':
            return float(floor(_number(args[0])))
        if name == 'ceiling':
            return float(ceil(_number(args[0])))
        if name == 'round':
            return float(floor(_number(args[0]) + .5))
        if name == 'name':
            target = args[0][0] if args and args[0] else node
            return text_type(target.name or '')
        if name == 'local-name':
            target = args[0][0] if args and args[0] else node
            return _qname_parts(target.name or '')[1]
        if name == 'namespace-uri':
            target = args[0][0] if args and args[0] else node
            return _qname_parts(target.name or '')[0] or ''
        return False
    return False


def _matches_name(node, test, namespaces):
    if test == 'node()':
        return node.kind in ('element', 'text', 'comment', 'pi')
    if test == 'text()':
        return node.kind == 'text'
    if test == 'comment()':
        return node.kind == 'comment'
    if test.startswith('processing-instruction'):
        if node.kind != 'pi':
            return False
        match = re.match(r'processing-instruction\((.*)\)$', test)
        if not match or not match.group(1):
            return True
        target = match.group(1).strip(' "\'')
        try:
            return node.data[0] == target
        except Exception:
            return False
    if node.kind not in ('element', 'attribute'):
        return False
    uri, local = _qname_parts(node.name)
    if test == '*':
        return True
    if test.endswith(':*'):
        prefix = test[:-2]
        wanted = namespaces.get(prefix)
        return uri == wanted
    if ':' in test:
        prefix, wanted_local = test.split(':', 1)
        wanted_uri = namespaces.get(prefix)
        return uri == wanted_uri and local == wanted_local
    if test.startswith('{'):
        return text_type(node.name) == test
    return uri is None and local == test


def _descendants(node):
    result = []
    for child in node.children:
        result.append(child)
        result.extend(_descendants(child))
    return result


def _split_path(source, error):
    steps = []
    current = []
    bracket = 0
    quote = None
    index = 0
    separator = None
    while index < len(source):
        char = source[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in ('"', "'"):
            quote = char
            current.append(char)
            index += 1
            continue
        if char == '[':
            bracket += 1
        elif char == ']':
            bracket -= 1
            if bracket < 0:
                raise error('Unbalanced predicate')
        if char == '/' and bracket == 0:
            if current:
                steps.append((separator, ''.join(current).strip()))
                current = []
            elif separator is not None:
                raise error('Empty location step')
            separator = '//' if index + 1 < len(source) and source[index + 1] == '/' else '/'
            index += 2 if separator == '//' else 1
            continue
        current.append(char)
        index += 1
    if quote or bracket:
        raise error('Unterminated path expression')
    if current:
        steps.append((separator, ''.join(current).strip()))
    elif separator is not None:
        raise error('Path cannot end in a slash')
    return steps


def _parse_step(text, separator, error):
    predicates = []
    base = []
    index = 0
    while index < len(text):
        if text[index] != '[':
            base.append(text[index])
            index += 1
            continue
        depth = 1
        quote = None
        end = index + 1
        while end < len(text) and depth:
            char = text[end]
            if quote:
                if char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
            end += 1
        if depth:
            raise error('Unterminated predicate')
        predicates.append(_ExpressionParser(text[index + 1:end - 1], error).parse())
        index = end
    base = ''.join(base).strip()
    if not base:
        raise error('Missing node test')
    axis = CHILD
    test = base
    if base == '.':
        axis, test = SELF, 'node()'
    elif base == '..':
        raise error('The parent axis is not supported')
    elif base.startswith('@'):
        axis, test = ATTRIBUTE, base[1:]
    elif '::' in base:
        axis_name, test = base.split('::', 1)
        axis = Axis.forname(axis_name)
        if axis is None:
            raise error('Unsupported axis %s' % axis_name)
    if separator == '//':
        return [(DESCENDANT_OR_SELF, 'node()', []), (axis, test, predicates)]
    return [(axis, test, predicates)]


class Path(object):
    def __init__(self, text, filename=None, lineno=-1):
        self.source = text_type(text)
        self.filename = filename
        self.lineno = lineno

        def error(message):
            return PathSyntaxError(message, filename, lineno, -1, self.source)

        if not self.source.strip():
            raise error('Empty path expression')
        parsed = _split_path(self.source.strip(), error)
        self.path = []
        for separator, step in parsed:
            self.path.extend(_parse_step(step, separator, error))
        self.strategies = [SimplePathStrategy, GenericStrategy]

    def __repr__(self):
        return 'Path(%r)' % self.source

    def test(self, ignore_context=False):
        return GenericStrategy(self.path).test(ignore_context)

    def _evaluate(self, root, namespaces, variables):
        current = [root]
        for axis, test, predicates in self.path:
            result = []
            for context in current:
                if axis == SELF:
                    candidates = [context]
                elif axis == ATTRIBUTE:
                    candidates = [_AttributeNode(name, value, context)
                                  for name, value in context.attrs.items()]
                elif axis == DESCENDANT:
                    candidates = _descendants(context)
                elif axis == DESCENDANT_OR_SELF:
                    candidates = [context] + _descendants(context)
                else:
                    candidates = list(context.children)
                candidates = [candidate for candidate in candidates
                              if _matches_name(candidate, test, namespaces)]
                for predicate in predicates:
                    size = len(candidates)
                    filtered = []
                    for position, candidate in enumerate(candidates, 1):
                        value = _eval_expr(predicate, candidate, position, size,
                                           namespaces, variables)
                        if isinstance(value, float):
                            accepted = position == int(value)
                        else:
                            accepted = _boolean(value)
                        if accepted:
                            filtered.append(candidate)
                    candidates = filtered
                result.extend(candidates)
            current = result
        return current

    def select(self, stream, namespaces=None, variables=None):
        namespaces = dict(namespaces or {})
        variables = dict(variables or {})
        events = list(stream)
        root = _build_tree(events)
        selected = self._evaluate(root, namespaces, variables)

        intervals = []
        point_events = set()
        synthetic = []
        for node in selected:
            if isinstance(node, _AttributeNode):
                synthetic.append((TEXT, node.data, node.pos))
            elif node.kind == 'element':
                intervals.append((node.start, node.end))
            elif node.event is not None:
                point_events.add(node.event)

        intervals.sort()
        merged = []
        for start, end in intervals:
            if end is None:
                end = start
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        def generate():
            interval_index = 0
            for index, event in enumerate(events):
                while interval_index < len(merged) and \
                        index > merged[interval_index][1]:
                    interval_index += 1
                included = interval_index < len(merged) and \
                    merged[interval_index][0] <= index <= merged[interval_index][1]
                if included or index in point_events:
                    yield event
            for event in synthetic:
                yield event

        return Stream(generate())