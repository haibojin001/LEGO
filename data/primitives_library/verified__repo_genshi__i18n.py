from gettext import NullTranslations
import os
import re
from functools import partial
from types import FunctionType

from genshi.compat import ast, string_types, text_type, IS_PYTHON2, _ast_Str, _ast_Str_value
from genshi.core import Attrs, Namespace, QName, START, END, TEXT, XML_NAMESPACE, _ensure, StreamEventKind
from genshi.template.base import DirectiveFactory, EXPR, SUB, _apply_directives
from genshi.template.directives import Directive, StripDirective
from genshi.template.markup import MarkupTemplate, EXEC

__all__ = ['Translator', 'extract']
__docformat__ = 'restructuredtext en'

I18N_NAMESPACE = Namespace('http://genshi.edgewall.org/i18n')

MSGBUF = StreamEventKind('MSGBUF')
SUB_START = StreamEventKind('SUB_START')
SUB_END = StreamEventKind('SUB_END')

GETTEXT_FUNCTIONS = (
    '_', 'gettext', 'ngettext', 'dgettext', 'dngettext',
    'ugettext', 'ungettext'
)

contexted = {
    None: 'pgettext',
    'gettext': 'pgettext',
    'ngettext': 'pngettext',
    'dgettext': 'dpgettext',
    'dngettext': 'dnpgettext'
}


def contextify(line, func, msg, comment, context):
    if context:
        ctx = context[0]
        func = contexted.get(func)
        if func is None:
            raise ValueError('failure, bogus extraction method')
        if isinstance(msg, tuple):
            msg = (ctx, msg[0], msg[1])
        else:
            msg = (ctx, msg)
    return line, func, msg, comment


class I18NDirective(Directive):

    def __call__(self, stream, directives, ctxt, **vars):
        return _apply_directives(stream, directives, ctxt, vars)


class ExtractableI18NDirective(I18NDirective):

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS,
                search_text=True, comment_stack=None, context_stack=None):
        raise NotImplementedError


class CommentDirective(I18NDirective):

    __slots__ = ['comment']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.comment = value


class _ContextDirective(I18NDirective):

    __slots__ = ['value']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.value = value.strip()

    def _enter(self, ctxt, key):
        value = self.value
        if value.startswith('${') and value.endswith('}'):
            name = value[2:-1].strip()
            value = ctxt.get(name)
        if hasattr(ctxt, 'push'):
            ctxt.push({key: value})
            return True
        old = ctxt.get(key)
        ctxt[key] = value
        return old

    def _leave(self, ctxt, key, marker):
        if hasattr(ctxt, 'pop'):
            ctxt.pop()
        elif marker is None:
            ctxt.pop(key, None)
        else:
            ctxt[key] = marker


class DomainDirective(_ContextDirective):

    def __call__(self, stream, directives, ctxt, **vars):
        marker = self._enter(ctxt, '_i18n.domain')
        try:
            for event in _apply_directives(stream, directives, ctxt, vars):
                yield event
        finally:
            self._leave(ctxt, '_i18n.domain', marker)


class ContextDirective(_ContextDirective):

    def __call__(self, stream, directives, ctxt, **vars):
        marker = self._enter(ctxt, '_i18n.context')
        try:
            for event in _apply_directives(stream, directives, ctxt, vars):
                yield event
        finally:
            self._leave(ctxt, '_i18n.context', marker)


class MessageBuffer:

    _sub_re = re.compile(r'%\(([^)]+)\)[#0 +\-]?\d*(?:\.\d+)?[diouxXeEfFgGcrs]|'
                         r'\[(\d+):|(\])')

    def __init__(self, directive=None):
        self.directive = directive
        self.events = []
        self._starts = []
        self._tags = {}
        self._expressions = {}
        self._next_tag = 1

    def append(self, kind, data, pos):
        self.events.append((kind, data, pos))

    def _expression_name(self, data):
        source = getattr(data, 'source', None)
        if source is None:
            source = text_type(data)
        return source.strip()

    def format(self):
        pieces = []
        starts = []
        tagno = 0
        for kind, data, pos in self.events:
            if kind is TEXT:
                pieces.append(text_type(data))
            elif kind is EXPR:
                source = self._expression_name(data)
                pieces.append('%%(%s)s' % source)
            elif kind is START:
                tagno += 1
                starts.append(tagno)
                pieces.append('[%d:' % tagno)
            elif kind is END:
                if starts:
                    starts.pop()
                    pieces.append(']')
        return ''.join(pieces).strip()

    def _maps(self):
        tags = {}
        expressions = {}
        stack = []
        number = 0
        for event in self.events:
            kind, data, pos = event
            if kind is START:
                number += 1
                tags[number] = [event, None]
                stack.append(number)
            elif kind is END and stack:
                tags[stack.pop()][1] = event
            elif kind is EXPR:
                expressions[self._expression_name(data)] = event
        return tags, expressions

    def translate(self, message):
        tags, expressions = self._maps()
        pos = self.events[0][2] if self.events else (None, -1, -1)
        cursor = 0
        opened = []

        for match in self._sub_re.finditer(message):
            if match.start() > cursor:
                yield TEXT, message[cursor:match.start()], pos
            token = match.group(0)
            name, tag, closing = match.groups()
            if name is not None:
                event = expressions.get(name)
                if event is not None:
                    yield event
                else:
                    yield TEXT, token, pos
            elif tag is not None:
                number = int(tag)
                pair = tags.get(number)
                if pair is not None:
                    opened.append(number)
                    yield pair[0]
                else:
                    yield TEXT, token, pos
            elif closing:
                if opened:
                    number = opened.pop()
                    event = tags.get(number)
                    if event is not None and event[1] is not None:
                        yield event[1]
                else:
                    yield TEXT, token, pos
            cursor = match.end()

        if cursor < len(message):
            yield TEXT, message[cursor:], pos


class MsgDirective(ExtractableI18NDirective):

    __slots__ = ['params', 'lineno']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.params = [item.strip() for item in value.split(',') if item.strip()]
        self.lineno = lineno

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if isinstance(value, dict):
            value = value.get('params', '')
        return super(MsgDirective, cls).attach(
            template, stream, text_type(value).strip(), namespaces, pos
        )

    def _gettext(self, ctxt):
        domain = ctxt.get('_i18n.domain')
        context = ctxt.get('_i18n.context')
        if domain and context:
            func = ctxt.get('_i18n.dpgettext')
            assert callable(func), 'No domain/context gettext function passed'
            return lambda message: func(domain, context, message)
        if domain:
            func = ctxt.get('_i18n.dgettext')
            assert callable(func), 'No domain gettext function passed'
            return lambda message: func(domain, message)
        if context:
            func = ctxt.get('_i18n.pgettext')
            assert callable(func), 'No context gettext function passed'
            return lambda message: func(context, message)
        return ctxt.get('_i18n.gettext', lambda value: value)

    def __call__(self, stream, directives, ctxt, **vars):
        gettext = self._gettext(ctxt)

        def generate():
            iterator = iter(stream)
            try:
                first = next(iterator)
            except StopIteration:
                return
            buffer = MessageBuffer(self)
            if first[0] is START:
                yield first
            else:
                buffer.append(*first)

            previous = None
            for event in iterator:
                if previous is not None:
                    buffer.append(*previous)
                previous = event

            if previous is not None and previous[0] is END:
                end = previous
            else:
                end = None
                if previous is not None:
                    buffer.append(*previous)

            for event in buffer.translate(gettext(buffer.format())):
                yield event
            if end is not None:
                yield end

        return _apply_directives(generate(), directives, ctxt, vars)

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS,
                search_text=True, comment_stack=None, context_stack=None):
        comment_stack = comment_stack or []
        context_stack = context_stack or []
        buffer = MessageBuffer(self)
        iterator = iter(stream)
        try:
            previous = next(iterator)
        except StopIteration:
            return
        strip_outer = previous[0] is START
        if strip_outer:
            for item in translator._extract_attrs(previous, gettext_functions,
                                                  search_text):
                yield item
            try:
                previous = next(iterator)
            except StopIteration:
                previous = None

        if previous is not None:
            for event in iterator:
                if event[0] is START:
                    for item in translator._extract_attrs(event, gettext_functions,
                                                          search_text):
                        yield item
                buffer.append(*previous)
                previous = event
            if previous is not None and not strip_outer:
                buffer.append(*previous)

        yield contextify(self.lineno, None, buffer.format(),
                         comment_stack[-1:], context_stack[-1:])


class ChooseBranchDirective(ExtractableI18NDirective):

    __slots__ = ['lineno']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.lineno = lineno

    def __call__(self, stream, directives, ctxt, **vars):
        return _apply_directives(stream, directives, ctxt, vars)

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS,
                search_text=True, comment_stack=None, context_stack=None):
        buffer = MessageBuffer(self)
        for event in stream:
            buffer.append(*event)
        yield buffer.format()


class SingularDirective(ChooseBranchDirective):
    pass


class PluralDirective(ChooseBranchDirective):
    pass


class ChooseDirective(ExtractableI18NDirective):

    __slots__ = ['count', 'lineno']

    def __init__(self, value, template=None, namespaces=None, lineno=-1,
                 offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.count = value.strip()
        self.lineno = lineno

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if isinstance(value, dict):
            value = value.get('count', '')
        return super(ChooseDirective, cls).attach(template, stream, value,
                                                  namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        return _apply_directives(stream, directives, ctxt, vars)

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS,
                search_text=True, comment_stack=None, context_stack=None):
        comment_stack = comment_stack or []
        context_stack = context_stack or []
        singular = None
        plural = None

        for directive, branch in translator._substreams(stream):
            if isinstance(directive, SingularDirective):
                singular = translator._branch_message(branch)
            elif isinstance(directive, PluralDirective):
                plural = translator._branch_message(branch)

        if singular is None:
            singular = ''
        if plural is None:
            plural = singular

        yield contextify(self.lineno, 'ngettext', (singular, plural),
                         comment_stack[-1:], context_stack[-1:])


class Translator:

    DEFAULT_IGNORE_TAGS = frozenset((
        QName('script'), QName('style'), QName('code')
    ))
    DEFAULT_INCLUDE_ATTRS = frozenset((
        QName('abbr'), QName('alt'), QName('label'), QName('prompt'),
        QName('standby'), QName('summary'), QName('title'), QName('value')
    ))

    def __init__(self, translations=None, ignore_tags=DEFAULT_IGNORE_TAGS,
                 include_attrs=DEFAULT_INCLUDE_ATTRS, extract_text=True):
        self.translations = translations or NullTranslations()
        self.ignore_tags = set(ignore_tags or ())
        self.include_attrs = set(include_attrs or ())
        self.extract_text = extract_text

        self.gettext = getattr(self.translations, 'gettext',
                               lambda text: text)
        self.ngettext = getattr(self.translations, 'ngettext',
                                lambda singular, plural, n: singular if n == 1 else plural)
        self.dgettext = getattr(self.translations, 'dgettext',
                                lambda domain, text: self.gettext(text))
        self.dngettext = getattr(self.translations, 'dngettext',
                                 lambda domain, singular, plural, n:
                                 self.ngettext(singular, plural, n))
        self.pgettext = getattr(self.translations, 'pgettext',
                                lambda context, text: self.gettext(text))
        self.npgettext = getattr(self.translations, 'npgettext',
                                 lambda context, singular, plural, n:
                                 self.ngettext(singular, plural, n))
        self.dpgettext = getattr(self.translations, 'dpgettext',
                                 lambda domain, context, text:
                                 self.dgettext(domain, text))
        self.dnpgettext = getattr(self.translations, 'dnpgettext',
                                  lambda domain, context, singular, plural, n:
                                  self.dngettext(domain, singular, plural, n))

    def setup(self, template):
        directives = {
            'msg': MsgDirective,
            'choose': ChooseDirective,
            'singular': SingularDirective,
            'plural': PluralDirective,
            'comment': CommentDirective,
            'domain': DomainDirective,
            'context': ContextDirective
        }
        try:
            template.add_directives(I18N_NAMESPACE, directives)
        except (AttributeError, TypeError):
            try:
                template.add_directives(I18N_NAMESPACE, DirectiveFactory(directives))
            except (AttributeError, TypeError):
                pass
        if self not in template.filters:
            template.filters.insert(0, self)

    def __call__(self, stream, ctxt=None, **vars):
        if ctxt is None:
            return stream
        values = {
            '_i18n.gettext': self.gettext,
            '_i18n.ngettext': self.ngettext,
            '_i18n.dgettext': self.dgettext,
            '_i18n.dngettext': self.dngettext,
            '_i18n.pgettext': self.pgettext,
            '_i18n.npgettext': self.npgettext,
            '_i18n.dpgettext': self.dpgettext,
            '_i18n.dnpgettext': self.dnpgettext
        }
        if hasattr(ctxt, 'push'):
            ctxt.push(values)
            try:
                for event in stream:
                    yield event
            finally:
                ctxt.pop()
        else:
            old = dict((key, ctxt.get(key)) for key in values)
            ctxt.update(values)
            try:
                for event in stream:
                    yield event
            finally:
                for key, value in old.items():
                    if value is None:
                        ctxt.pop(key, None)
                    else:
                        ctxt[key] = value

    def _string_value(self, node):
        if isinstance(node, _ast_Str):
            return _ast_Str_value(node)
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            if isinstance(node.value, string_types):
                return node.value
        return None

    def _function_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_code(self, source, lineno, gettext_functions, comments,
                      contexts):
        if not source:
            return
        try:
            tree = ast.parse(source, mode='eval')
        except (SyntaxError, TypeError, ValueError):
            try:
                tree = ast.parse(source)
            except (SyntaxError, TypeError, ValueError):
                return

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = self._function_name(node.func)
            if func not in gettext_functions:
                continue
            args = list(node.args)
            if not args:
                continue
            values = [self._string_value(arg) for arg in args]
            if func in ('ngettext', 'ungettext', 'dngettext'):
                if len(values) < 2 or values[0] is None or values[1] is None:
                    continue
                msg = (values[0], values[1])
            elif func in ('dgettext',):
                if len(values) < 2 or values[1] is None:
                    continue
                msg = values[1]
            elif func in ('dngettext',):
                if len(values) < 3 or values[1] is None or values[2] is None:
                    continue
                msg = (values[1], values[2])
            else:
                if values[0] is None:
                    continue
                msg = values[0]
            line = lineno + getattr(node, 'lineno', 1) - 1
            yield contextify(line, func, msg, comments[-1:], contexts[-1:])

    def _extract_attrs(self, event, gettext_functions, search_text=True):
        kind, data, pos = event
        if kind is not START:
            return
        try:
            tag, attrs = data
        except (TypeError, ValueError):
            return
        if tag in self.ignore_tags:
            return
        for name, value in attrs:
            if name not in self.include_attrs:
                continue
            source = getattr(value, 'source', None)
            if source:
                for item in self._extract_code(source, pos[1], gettext_functions,
                                               [], []):
                    yield item
            elif search_text and isinstance(value, string_types):
                value = value.strip()
                if value:
                    yield pos[1], None, value, []

    def _substreams(self, stream):
        for event in stream:
            if event[0] is not SUB:
                continue
            data = event[1]
            if not isinstance(data, tuple) or len(data) != 2:
                continue
            first, second = data
            if isinstance(first, (list, tuple)):
                directives, child = first, second
            else:
                child, directives = first, second
            directive = None
            for item in directives:
                if isinstance(item, Directive):
                    directive = item
                    break
                if isinstance(item, tuple) and item and isinstance(item[0], Directive):
                    directive = item[0]
                    break
            yield directive, child

    def _branch_message(self, stream):
        buffer = MessageBuffer()
        for event in stream:
            buffer.append(*event)
        return buffer.format()

    def extract(self, stream, gettext_functions=GETTEXT_FUNCTIONS,
                search_text=True, comment_stack=None, context_stack=None):
        comments = list(comment_stack or [])
        contexts = list(context_stack or [])

        for kind, data, pos in stream:
            if kind is TEXT:
                if search_text and self.extract_text:
                    text = text_type(data).strip()
                    if text:
                        yield contextify(pos[1], None, text, comments[-1:],
                                         contexts[-1:])
            elif kind is START:
                for item in self._extract_attrs((kind, data, pos),
                                               gettext_functions, search_text):
                    yield item
            elif kind is EXPR:
                source = getattr(data, 'source', None)
                if source:
                    for item in self._extract_code(source, pos[1],
                                                   gettext_functions, comments,
                                                   contexts):
                        yield item
            elif kind is SUB:
                subdata = data
                if not isinstance(subdata, tuple) or len(subdata) != 2:
                    continue
                first, second = subdata
                if isinstance(first, (list, tuple)):
                    directives, child = first, second
                else:
                    child, directives = first, second

                active = None
                for item in directives:
                    candidate = item[0] if isinstance(item, tuple) else item
                    if isinstance(candidate, CommentDirective):
                        comments.append(candidate.comment)
                    elif isinstance(candidate, ContextDirective):
                        contexts.append(candidate.value)
                    elif isinstance(candidate, ExtractableI18NDirective):
                        active = candidate

                if active is not None:
                    for item in active.extract(self, child, gettext_functions,
                                               search_text, comments, contexts):
                        yield item
                else:
                    for item in self.extract(child, gettext_functions,
                                             search_text, comments, contexts):
                        yield item

                for item in reversed(directives):
                    candidate = item[0] if isinstance(item, tuple) else item
                    if isinstance(candidate, CommentDirective) and comments:
                        comments.pop()
                    elif isinstance(candidate, ContextDirective) and contexts:
                        contexts.pop()


def extract(fileobj, keywords=GETTEXT_FUNCTIONS, comment_tags=(),
            options=None):
    options = options or {}
    encoding = options.get('encoding', 'utf-8')
    source = fileobj.read()
    if isinstance(source, bytes):
        source = source.decode(encoding)
    filename = getattr(fileobj, 'name', None)
    template = MarkupTemplate(source, filename=filename, encoding=encoding)
    translator = Translator()
    translator.setup(template)
    for message in translator.extract(template.stream, keywords):
        yield message