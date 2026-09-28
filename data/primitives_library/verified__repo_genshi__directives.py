from genshi.compat import add_metaclass, text_type
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
                                 EXPR, _apply_directives, _eval_expr
from genshi.template.eval import Expression, _ast, _parse

__all__ = ['AttrsDirective', 'ChooseDirective', 'ContentDirective',
           'DefDirective', 'ForDirective', 'IfDirective', 'MatchDirective',
           'OtherwiseDirective', 'ReplaceDirective', 'StripDirective',
           'WhenDirective', 'WithDirective']
__docformat__ = 'restructuredtext en'


class DirectiveMeta(type):

    def __new__(mcls, name, bases, namespace):
        namespace['tagname'] = name.lower().replace('directive', '')
        return type.__new__(mcls, name, bases, namespace)


@add_metaclass(DirectiveMeta)
class Directive(object):

    __slots__ = ['expr']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        self.expr = self._parse_expr(value, template, lineno, offset)

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        return cls(value, template, namespaces, *pos[1:]), stream

    def __call__(self, stream, directives, ctxt, **vars):
        raise NotImplementedError

    def __repr__(self):
        value = ''
        if getattr(self, 'expr', None) is not None:
            value = ' "%s"' % self.expr.source
        return '<%s%s>' % (type(self).__name__, value)

    @classmethod
    def _parse_expr(cls, expr, template, lineno=-1, offset=-1):
        try:
            filepath = getattr(template, 'filepath', None)
            lookup = getattr(template, 'lookup', None)
            return expr and Expression(expr, filepath, lineno,
                                       lookup=lookup) or None
        except SyntaxError as err:
            err.msg += ' in expression "%s" of "%s" directive' % (
                expr, cls.tagname
            )
            raise TemplateSyntaxError(err, getattr(template, 'filepath', None),
                                      lineno, offset + (err.offset or 0))


def _assignment(ast):
    def names(node):
        if isinstance(node, _ast.Tuple):
            return tuple(names(child) for child in node.elts)
        if isinstance(node, _ast.Name):
            return node.id

    def assign(data, value, target=names(ast)):
        if type(target) is tuple:
            for index, name in enumerate(target):
                assign(data, value[index], name)
        else:
            data[target] = value
    return assign


def _attrs_value(value, name, default=None):
    if value is None:
        return default
    try:
        return value.get(name, default)
    except AttributeError:
        pass
    try:
        return dict(value).get(name, default)
    except (TypeError, ValueError):
        return default


def _syntax_error(template, lineno, offset, message):
    error = SyntaxError(message)
    error.msg = message
    raise TemplateSyntaxError(error, getattr(template, 'filepath', None),
                              lineno, offset)


def _parse_statement(source, template, lineno, offset, directive):
    try:
        return compile(source, getattr(template, 'filepath', None) or '<string>',
                       'exec', getattr(_ast, 'PyCF_ONLY_AST', 0))
    except SyntaxError as err:
        err.msg += ' in "%s" directive' % directive
        raise TemplateSyntaxError(err, getattr(template, 'filepath', None),
                                  lineno, offset + (err.offset or 0))


class AttrsDirective(Directive):

    __slots__ = []

    def __call__(self, stream, directives, ctxt, **vars):
        def generate():
            kind, (tag, attrib), pos = next(stream)
            attrs = _eval_expr(self.expr, ctxt, vars)
            if attrs:
                if isinstance(attrs, Stream):
                    try:
                        attrs = next(iter(attrs))
                    except StopIteration:
                        attrs = []
                elif not isinstance(attrs, list):
                    attrs = attrs.items()
                attrib |= [(QName(name),
                            text_type(value).strip() if value is not None
                            else None)
                           for name, value in attrs]
            yield kind, (tag, attrib), pos
            for event in stream:
                yield event
        return _apply_directives(generate(), directives, ctxt, vars)


class ContentDirective(Directive):

    __slots__ = []

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            raise TemplateSyntaxError(
                'The content directive can not be used as an element',
                template.filepath, *pos[1:]
            )
        expr = cls._parse_expr(value, template, *pos[1:])
        return None, [stream[0], (EXPR, expr, pos), stream[-1]]


class DefDirective(Directive):

    __slots__ = ['name', 'args', 'defaults', 'vararg', 'kwarg',
                 'kwonlyargs', 'kwdefaults']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        node = _parse_statement('def %s:\n pass' % value, template, lineno,
                                offset, self.tagname).body[0]
        self.name = node.name
        arguments = node.args
        self.args = [arg.arg if hasattr(arg, 'arg') else arg.id
                     for arg in arguments.args]
        self.vararg = (arguments.vararg.arg if hasattr(arguments.vararg, 'arg')
                       else arguments.vararg.id if arguments.vararg else None)
        self.kwarg = (arguments.kwarg.arg if hasattr(arguments.kwarg, 'arg')
                      else arguments.kwarg.id if arguments.kwarg else None)
        self.kwonlyargs = [arg.arg if hasattr(arg, 'arg') else arg.id
                           for arg in getattr(arguments, 'kwonlyargs', [])]
        self.defaults = list(arguments.defaults)
        self.kwdefaults = list(getattr(arguments, 'kw_defaults', []))
        self.expr = None

    def __call__(self, stream, directives, ctxt, **vars):
        args = self.args[:]
        default_nodes = self.defaults[:]
        kwonly = self.kwonlyargs[:]
        kwdefault_nodes = self.kwdefaults[:]
        name = self.name
        body = list(stream)
        remaining = list(directives)

        def evaluate_node(node):
            try:
                source = _ast.unparse(node)
            except AttributeError:
                source = None
            if source is None:
                return None
            return _eval_expr(Expression(source), ctxt, vars)

        def function(*values, **keywords):
            data = {}
            defaults = [evaluate_node(node) for node in default_nodes]
            first_default = len(args) - len(defaults)

            if len(values) > len(args) and not self.vararg:
                raise TypeError('%s() takes at most %d argument%s (%d given)' %
                                (name, len(args),
                                 '' if len(args) == 1 else 's', len(values)))

            for index, value in enumerate(values[:len(args)]):
                data[args[index]] = value

            if self.vararg:
                data[self.vararg] = tuple(values[len(args):])

            for key, value in keywords.items():
                if key in data:
                    raise TypeError("%s() got multiple values for keyword "
                                    "argument '%s'" % (name, key))
                if key in args or key in kwonly:
                    data[key] = value
                elif self.kwarg:
                    data.setdefault(self.kwarg, {})[key] = value
                else:
                    raise TypeError("%s() got an unexpected keyword argument "
                                    "'%s'" % (name, key))

            if self.kwarg and self.kwarg not in data:
                data[self.kwarg] = {}

            for index, argname in enumerate(args):
                if argname not in data:
                    if index >= first_default:
                        data[argname] = defaults[index - first_default]
                    else:
                        raise TypeError("%s() missing required argument '%s'" %
                                        (name, argname))

            for index, argname in enumerate(kwonly):
                if argname not in data:
                    default = kwdefault_nodes[index]
                    if default is None:
                        raise TypeError("%s() missing required keyword-only "
                                        "argument '%s'" % (name, argname))
                    data[argname] = evaluate_node(default)

            def generate():
                ctxt.push(data)
                try:
                    for event in _apply_directives(iter(body), remaining,
                                                   ctxt, vars):
                        yield event
                finally:
                    ctxt.pop()

            return Stream(generate())

        ctxt[name] = function
        return iter(())


class ForDirective(Directive):

    __slots__ = ['assign']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        statement = _parse_statement('for %s:\n pass' % value, template,
                                     lineno, offset, self.tagname)
        node = statement.body[0]
        self.assign = _assignment(node.target)
        try:
            expression = _ast.unparse(node.iter)
        except AttributeError:
            expression = value.split(' in ', 1)[-1]
        self.expr = self._parse_expr(expression, template, lineno, offset)

    def __call__(self, stream, directives, ctxt, **vars):
        events = list(stream)

        def generate():
            for item in _eval_expr(self.expr, ctxt, vars):
                ctxt.push({})
                try:
                    self.assign(ctxt, item)
                    for event in _apply_directives(iter(events), directives,
                                                   ctxt, vars):
                        yield event
                finally:
                    ctxt.pop()

        return generate()


class IfDirective(Directive):

    __slots__ = []

    def __call__(self, stream, directives, ctxt, **vars):
        if _eval_expr(self.expr, ctxt, vars):
            return _apply_directives(stream, directives, ctxt, vars)
        return iter(())


class ChooseDirective(Directive):

    __slots__ = []

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict or not isinstance(value, text_type):
            value = _attrs_value(value, 'test')
        return cls(value, template, namespaces, *pos[1:]), stream

    def __call__(self, stream, directives, ctxt, **vars):
        state = {
            'value': _eval_expr(self.expr, ctxt, vars) if self.expr else None,
            'selected': False,
            'has_test': self.expr is not None
        }
        localvars = vars.copy()
        localvars['__genshi_choose__'] = state
        return _apply_directives(stream, directives, ctxt, localvars)


class WhenDirective(Directive):

    __slots__ = []

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict or not isinstance(value, text_type):
            value = _attrs_value(value, 'test')
        return cls(value, template, namespaces, *pos[1:]), stream

    def __call__(self, stream, directives, ctxt, **vars):
        state = vars.get('__genshi_choose__')
        if state is None:
            raise TemplateRuntimeError('when directive can only be used '
                                       'inside a choose directive')
        if state['selected']:
            return iter(())
        value = _eval_expr(self.expr, ctxt, vars)
        matched = (state['value'] == value if state['has_test'] else bool(value))
        if matched:
            state['selected'] = True
            return _apply_directives(stream, directives, ctxt, vars)
        return iter(())


class OtherwiseDirective(Directive):

    __slots__ = []

    def __init__(self, value=None, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        self.expr = None

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        return cls(None, template, namespaces, *pos[1:]), stream

    def __call__(self, stream, directives, ctxt, **vars):
        state = vars.get('__genshi_choose__')
        if state is None:
            raise TemplateRuntimeError('otherwise directive can only be used '
                                       'inside a choose directive')
        if state['selected']:
            return iter(())
        state['selected'] = True
        return _apply_directives(stream, directives, ctxt, vars)


class MatchDirective(Directive):

    __slots__ = ['path', 'once', 'buffer']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1, once=False, buffer=True):
        try:
            self.path = Path(value, filename=getattr(template, 'filepath', None),
                             lineno=lineno)
        except TypeError:
            self.path = Path(value)
        self.expr = None
        self.once = once
        self.buffer = buffer

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if isinstance(value, text_type):
            path = value
            once = False
            buffer = True
        else:
            path = _attrs_value(value, 'path')
            once = _attrs_value(value, 'once', False)
            buffer = _attrs_value(value, 'buffer', True)
            once = text_type(once).lower() in ('1', 'true', 'yes', 'on')
            buffer = text_type(buffer).lower() not in ('0', 'false', 'no', 'off')
        if not path:
            raise TemplateSyntaxError('The match directive requires a path '
                                      'attribute', template.filepath, *pos[1:])
        return cls(path, template, namespaces, *pos[1:], once=once,
                   buffer=buffer), stream

    def __call__(self, stream, directives, ctxt, **vars):
        ctxt._match_templates.append((self, stream, directives))
        return iter(())


class ReplaceDirective(Directive):

    __slots__ = []

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            raise TemplateSyntaxError(
                'The replace directive can not be used as an element',
                template.filepath, *pos[1:]
            )
        expr = cls._parse_expr(value, template, *pos[1:])
        return None, [(EXPR, expr, pos)]


class StripDirective(Directive):

    __slots__ = []

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        self.expr = self._parse_expr(value, template, lineno, offset) \
                    if value else None

    def __call__(self, stream, directives, ctxt, **vars):
        if self.expr is not None and not _eval_expr(self.expr, ctxt, vars):
            return _apply_directives(stream, directives, ctxt, vars)

        events = iter(stream)

        def generate():
            try:
                next(events)
            except StopIteration:
                return
            buffered = list(events)
            if buffered:
                buffered.pop()
            for event in buffered:
                yield event

        return _apply_directives(generate(), directives, ctxt, vars)


class WithDirective(Directive):

    __slots__ = ['assignments']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        self.expr = None
        self.assignments = []
        source = ';\n'.join(part for part in value.split(';') if part.strip())
        statement = _parse_statement(source, template, lineno, offset,
                                     self.tagname)
        for node in statement.body:
            if not isinstance(node, _ast.Assign) or len(node.targets) != 1:
                _syntax_error(template, lineno, offset,
                              'Invalid assignment in with directive')
            target = node.targets[0]
            try:
                expression = _ast.unparse(node.value)
            except AttributeError:
                expression = None
            if expression is None:
                _syntax_error(template, lineno, offset,
                              'Invalid assignment in with directive')
            self.assignments.append((
                _assignment(target),
                self._parse_expr(expression, template, lineno, offset)
            ))

    def __call__(self, stream, directives, ctxt, **vars):
        def generate():
            ctxt.push({})
            try:
                for assign, expression in self.assignments:
                    assign(ctxt, _eval_expr(expression, ctxt, vars))
                for event in _apply_directives(stream, directives, ctxt, vars):
                    yield event
            finally:
                ctxt.pop()

        return generate()