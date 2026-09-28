from itertools import chain
import re

from genshi.compat import string_types, text_type
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
                        START_CDATA, END_CDATA, PI, COMMENT, XML_NAMESPACE

__all__ = ['encode', 'get_serializer', 'DocType', 'XMLSerializer',
           'XHTMLSerializer', 'HTMLSerializer', 'TextSerializer']
__docformat__ = 'restructuredtext en'


def encode(iterator, method='xml', encoding=None, out=None):
    if encoding is None:
        convert = lambda value: value
    else:
        errors = 'replace'
        if method != 'text' and not isinstance(method, TextSerializer):
            errors = 'xmlcharrefreplace'
        convert = lambda value: value.encode(encoding, errors)

    if out is None:
        return convert(''.join(iterator))

    for chunk in iterator:
        out.write(convert(chunk))


def get_serializer(method='xml', **kwargs):
    if isinstance(method, string_types):
        serializers = {
            'xml': XMLSerializer,
            'xhtml': XHTMLSerializer,
            'html': HTMLSerializer,
            'text': TextSerializer
        }
        method = serializers[method.lower()]
    return method(**kwargs)


def _prepare_cache(use_cache=True):
    cache = {}
    if use_cache:
        def emit(kind, input, output):
            try:
                cache[kind, input] = output
            except TypeError:
                pass
            return output

        def get(key):
            try:
                return cache.get(key)
            except TypeError:
                return None
    else:
        def emit(kind, input, output):
            return output

        def get(key):
            return None
    return emit, get, cache


class DocType(object):
    HTML_STRICT = (
        'html', '-//W3C//DTD HTML 4.01//EN',
        'http://www.w3.org/TR/html4/strict.dtd'
    )
    HTML_TRANSITIONAL = (
        'html', '-//W3C//DTD HTML 4.01 Transitional//EN',
        'http://www.w3.org/TR/html4/loose.dtd'
    )
    HTML_FRAMESET = (
        'html', '-//W3C//DTD HTML 4.01 Frameset//EN',
        'http://www.w3.org/TR/html4/frameset.dtd'
    )
    HTML = HTML_STRICT

    HTML5 = ('html', None, None)

    XHTML_STRICT = (
        'html', '-//W3C//DTD XHTML 1.0 Strict//EN',
        'http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd'
    )
    XHTML_TRANSITIONAL = (
        'html', '-//W3C//DTD XHTML 1.0 Transitional//EN',
        'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd'
    )
    XHTML_FRAMESET = (
        'html', '-//W3C//DTD XHTML 1.0 Frameset//EN',
        'http://www.w3.org/TR/xhtml1/DTD/xhtml1-frameset.dtd'
    )
    XHTML = XHTML_STRICT

    XHTML11 = (
        'html', '-//W3C//DTD XHTML 1.1//EN',
        'http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd'
    )

    SVG_FULL = (
        'svg', '-//W3C//DTD SVG 1.1//EN',
        'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'
    )
    SVG_BASIC = (
        'svg', '-//W3C//DTD SVG Basic 1.1//EN',
        'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11-basic.dtd'
    )
    SVG_TINY = (
        'svg', '-//W3C//DTD SVG Tiny 1.1//EN',
        'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11-tiny.dtd'
    )
    SVG = SVG_FULL

    @classmethod
    def get(cls, name):
        return {
            'html': cls.HTML, 'html-strict': cls.HTML_STRICT,
            'html-transitional': cls.HTML_TRANSITIONAL,
            'html-frameset': cls.HTML_FRAMESET,
            'html5': cls.HTML5,
            'xhtml': cls.XHTML, 'xhtml-strict': cls.XHTML_STRICT,
            'xhtml-transitional': cls.XHTML_TRANSITIONAL,
            'xhtml-frameset': cls.XHTML_FRAMESET,
            'xhtml11': cls.XHTML11,
            'svg': cls.SVG, 'svg-full': cls.SVG_FULL,
            'svg-basic': cls.SVG_BASIC,
            'svg-tiny': cls.SVG_TINY
        }.get(name.lower())


def _qname_parts(name):
    namespace = getattr(name, 'namespace', None)
    localname = getattr(name, 'localname', None)
    if localname is not None:
        return namespace, text_type(localname)

    value = text_type(name)
    if value.startswith('{'):
        closing = value.find('}')
        if closing >= 0:
            return value[1:closing], value[closing + 1:]
    return None, value


def _escape_text(value):
    return text_type(escape(value, False))


def _escape_attr(value):
    return text_type(escape(value, True))


def _doctype_text(value):
    name, pubid, sysid = value
    if pubid:
        if sysid:
            return '<!DOCTYPE %s PUBLIC "%s" "%s">' % (name, pubid, sysid)
        return '<!DOCTYPE %s PUBLIC "%s">' % (name, pubid)
    if sysid:
        return '<!DOCTYPE %s SYSTEM "%s">' % (name, sysid)
    return '<!DOCTYPE %s>' % name


class EmptyTagFilter(object):
    def __call__(self, stream):
        stream = iter(stream)
        previous = None

        for event in stream:
            if previous is None:
                previous = event
                continue

            if (previous[0] is START and event[0] is END and
                    _qname_parts(previous[1][0]) == _qname_parts(event[1])):
                tag, attrs = previous[1][:2]
                yield START, (tag, attrs, True), previous[2]
                previous = None
            else:
                yield previous
                previous = event

        if previous is not None:
            yield previous


class WhitespaceFilter(object):
    _whitespace = re.compile(r'\s+')

    def __init__(self, preserve_space=frozenset()):
        self.preserve_space = frozenset(text_type(item) for item in preserve_space)

    def __call__(self, stream):
        preserve = 0
        stack = []

        for kind, data, pos in stream:
            if kind is START:
                tag = text_type(data[0])
                attrs = data[1]
                inherited = preserve
                xml_space = None
                for name, value in attrs:
                    namespace, localname = _qname_parts(name)
                    if ((namespace == XML_NAMESPACE and localname == 'space') or
                            text_type(name) == 'xml:space'):
                        xml_space = text_type(value)
                        break
                active = tag in self.preserve_space or xml_space == 'preserve'
                stack.append((inherited, xml_space))
                if active:
                    preserve += 1
                elif xml_space == 'default':
                    preserve = 0
                yield kind, data, pos

            elif kind is END:
                yield kind, data, pos
                if stack:
                    inherited, xml_space = stack.pop()
                    preserve = inherited

            elif kind is TEXT and not preserve:
                value = text_type(data).strip()
                if value:
                    yield kind, value, pos
            else:
                yield kind, data, pos


class NamespaceFlattener(object):
    def __init__(self, prefixes=None, cache=True):
        self.prefixes = dict(prefixes or {})
        self.cache = cache

    def __call__(self, stream):
        prefix_to_uri = {'xml': XML_NAMESPACE}
        uri_to_prefix = {XML_NAMESPACE: 'xml'}
        pending = []
        undo = []
        generated = [0]

        def preferred(uri, attribute=False):
            if uri == XML_NAMESPACE:
                return 'xml'

            prefix = self.prefixes.get(uri)
            if prefix is not None:
                prefix = text_type(prefix)
                if attribute and not prefix:
                    prefix = None
                if prefix is not None:
                    return prefix

            prefix = uri_to_prefix.get(uri)
            if prefix is not None and (prefix or not attribute):
                return prefix

            for candidate, mapped_uri in prefix_to_uri.items():
                if mapped_uri == uri and (candidate or not attribute):
                    return candidate

            while True:
                generated[0] += 1
                candidate = 'ns%d' % generated[0]
                if candidate not in prefix_to_uri:
                    return candidate

        def install(prefix, uri, declarations):
            prefix = text_type(prefix or '')
            uri = text_type(uri)
            old = prefix_to_uri.get(prefix)
            undo.append((prefix, old))
            prefix_to_uri[prefix] = uri
            if uri not in uri_to_prefix or not uri_to_prefix[uri]:
                uri_to_prefix[uri] = prefix
            declarations.append((prefix, uri))

        def lexical(name, attribute, declarations):
            uri, local = _qname_parts(name)
            if not uri:
                return local
            prefix = preferred(uri, attribute)
            if prefix_to_uri.get(prefix) != uri:
                install(prefix, uri, declarations)
            return '%s:%s' % (prefix, local) if prefix else local

        for kind, data, pos in stream:
            if kind is START_NS:
                prefix, uri = data
                wanted = self.prefixes.get(uri, prefix)
                pending.append((wanted, uri))
                continue

            if kind is END_NS:
                if undo:
                    prefix, old = undo.pop()
                    current = prefix_to_uri.pop(prefix, None)
                    if old is not None:
                        prefix_to_uri[prefix] = old
                    if current is not None and uri_to_prefix.get(current) == prefix:
                        replacement = None
                        for candidate, mapped_uri in prefix_to_uri.items():
                            if mapped_uri == current:
                                replacement = candidate
                                break
                        if replacement is None:
                            uri_to_prefix.pop(current, None)
                        else:
                            uri_to_prefix[current] = replacement
                continue

            if kind is START:
                tag = data[0]
                attrs = data[1]
                empty = len(data) > 2 and data[2]
                declarations = []
                for prefix, uri in pending:
                    if prefix == 'xml' and uri == XML_NAMESPACE:
                        continue
                    install(prefix, uri, declarations)
                pending = []

                tag = lexical(tag, False, declarations)
                output_attrs = []
                seen_declarations = set()
                for prefix, uri in declarations:
                    key = (prefix, uri)
                    if key in seen_declarations:
                        continue
                    seen_declarations.add(key)
                    output_attrs.append((
                        'xmlns' if not prefix else 'xmlns:%s' % prefix, uri
                    ))

                for name, value in attrs:
                    text_name = text_type(name)
                    if text_name == 'xmlns' or text_name.startswith('xmlns:'):
                        output_attrs.append((text_name, value))
                    else:
                        output_attrs.append((lexical(name, True, declarations), value))

                for prefix, uri in declarations:
                    key = ('xmlns' if not prefix else 'xmlns:%s' % prefix, uri)
                    if not any(name == key[0] for name, value in output_attrs):
                        output_attrs.insert(0, key)

                payload = (tag, Attrs(output_attrs), True) if empty else \
                          (tag, Attrs(output_attrs))
                yield kind, payload, pos
                continue

            if kind is END:
                yield kind, _qname_parts(data)[1] if not getattr(data, 'namespace', None) else lexical(data, False, []), pos
                continue

            yield kind, data, pos


class DocTypeInserter(object):
    def __init__(self, doctype):
        if isinstance(doctype, string_types):
            doctype = DocType.get(doctype)
        self.doctype = doctype

    def __call__(self, stream):
        inserted = False
        for kind, data, pos in stream:
            if kind is DOCTYPE:
                if not inserted and self.doctype:
                    yield DOCTYPE, self.doctype, pos
                    inserted = True
                elif not inserted:
                    yield kind, data, pos
                    inserted = True
                continue

            if kind is START and not inserted:
                if self.doctype:
                    yield DOCTYPE, self.doctype, pos
                inserted = True
            yield kind, data, pos


class XMLSerializer(object):
    _PRESERVE_SPACE = frozenset()

    def __init__(self, doctype=None, strip_whitespace=True,
                 namespace_prefixes=None, cache=True):
        self.filters = [EmptyTagFilter()]
        if strip_whitespace:
            self.filters.append(WhitespaceFilter(self._PRESERVE_SPACE))
        self.filters.append(NamespaceFlattener(prefixes=namespace_prefixes,
                                               cache=cache))
        if doctype:
            self.filters.append(DocTypeInserter(doctype))
        self.cache = cache

    def _prepare_cache(self):
        return _prepare_cache(self.cache)[:2]

    def __call__(self, stream):
        for filter_ in self.filters:
            stream = filter_(stream)

        emit, get = self._prepare_cache()
        have_decl = False
        have_doctype = False
        cdata = False

        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data[:2]
                empty = len(data) > 2 and data[2]
                cached = get((kind, data))
                if cached is None:
                    parts = ['<', text_type(tag)]
                    for name, value in attrs:
                        parts.extend((' ', text_type(name), '="',
                                      _escape_attr(value), '"'))
                    parts.append('/>' if empty else '>')
                    cached = emit(kind, data, ''.join(parts))
                yield cached

            elif kind is END:
                yield '</%s>' % text_type(data)

            elif kind is TEXT:
                if cdata:
                    yield text_type(data)
                else:
                    cached = get((kind, data))
                    if cached is None:
                        cached = emit(kind, data, _escape_text(data))
                    yield cached

            elif kind is XML_DECL:
                if not have_decl:
                    version, encoding, standalone = data
                    result = ['<?xml version="%s"' % version]
                    if encoding:
                        result.append(' encoding="%s"' % encoding)
                    if standalone is not None:
                        result.append(' standalone="%s"' %
                                      ('yes' if standalone else 'no'))
                    result.append('?>')
                    yield ''.join(result)
                    have_decl = True

            elif kind is DOCTYPE:
                if not have_doctype:
                    yield _doctype_text(data)
                    have_doctype = True

            elif kind is START_CDATA:
                cdata = True
                yield '<![CDATA['

            elif kind is END_CDATA:
                cdata = False
                yield ']]>'

            elif kind is PI:
                target, text = data
                yield '<?%s%s?>' % (target, (' ' + text) if text else '')

            elif kind is COMMENT:
                yield '<!--%s-->' % text_type(data)


class XHTMLSerializer(XMLSerializer):
    _PRESERVE_SPACE = frozenset((QName('pre'), QName('textarea')))
    _BOOLEAN_ATTRS = frozenset((
        'checked', 'compact', 'declare', 'defer', 'disabled', 'ismap',
        'multiple', 'nohref', 'noresize', 'noshade', 'nowrap', 'readonly',
        'selected'
    ))

    def __call__(self, stream):
        for filter_ in self.filters:
            stream = filter_(stream)

        emit, get = self._prepare_cache()
        have_decl = False
        have_doctype = False
        cdata = False

        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data[:2]
                empty = len(data) > 2 and data[2]
                cached = get((kind, data))
                if cached is None:
                    parts = ['<', text_type(tag)]
                    for name, value in attrs:
                        name = text_type(name)
                        if name.lower() in self._BOOLEAN_ATTRS and value:
                            value = name
                        parts.extend((' ', name, '="', _escape_attr(value), '"'))
                    parts.append(' />' if empty else '>')
                    cached = emit(kind, data, ''.join(parts))
                yield cached

            elif kind is END:
                yield '</%s>' % text_type(data)

            elif kind is TEXT:
                yield text_type(data) if cdata else _escape_text(data)

            elif kind is XML_DECL:
                if not have_decl:
                    version, encoding, standalone = data
                    result = ['<?xml version="%s"' % version]
                    if encoding:
                        result.append(' encoding="%s"' % encoding)
                    if standalone is not None:
                        result.append(' standalone="%s"' %
                                      ('yes' if standalone else 'no'))
                    result.append('?>')
                    yield ''.join(result)
                    have_decl = True

            elif kind is DOCTYPE:
                if not have_doctype:
                    yield _doctype_text(data)
                    have_doctype = True

            elif kind is START_CDATA:
                cdata = True
                yield '<![CDATA['

            elif kind is END_CDATA:
                cdata = False
                yield ']]>'

            elif kind is PI:
                target, text = data
                yield '<?%s%s?>' % (target, (' ' + text) if text else '')

            elif kind is COMMENT:
                yield '<!--%s-->' % text_type(data)


class HTMLSerializer(XMLSerializer):
    _PRESERVE_SPACE = frozenset((QName('pre'), QName('textarea'),
                                 QName('script'), QName('style')))
    _EMPTY_ELEMS = frozenset((
        'area', 'base', 'basefont', 'br', 'col', 'embed', 'frame', 'hr',
        'img', 'input', 'isindex', 'link', 'meta', 'param', 'source',
        'track', 'wbr'
    ))
    _BOOLEAN_ATTRS = frozenset((
        'allowfullscreen', 'async', 'autofocus', 'autoplay', 'checked',
        'compact', 'controls', 'declare', 'default', 'defer', 'disabled',
        'formnovalidate', 'hidden', 'inert', 'ismap', 'itemscope', 'loop',
        'multiple', 'muted', 'nohref', 'noresize', 'noshade', 'novalidate',
        'nowrap', 'open', 'readonly', 'required', 'reversed', 'selected',
        'truespeed', 'typemustmatch'
    ))
    _RAW_TEXT = frozenset(('script', 'style'))

    def __call__(self, stream):
        for filter_ in self.filters:
            stream = filter_(stream)

        raw_stack = []
        have_doctype = False

        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data[:2]
                tag = text_type(tag)
                lower_tag = tag.lower()
                parts = ['<', tag]
                for name, value in attrs:
                    name = text_type(name)
                    lower_name = name.lower()
                    if lower_name in self._BOOLEAN_ATTRS:
                        if value is False or value is None:
                            continue
                        parts.extend((' ', name))
                    else:
                        parts.extend((' ', name, '="', _escape_attr(value), '"'))
                parts.append('>')
                yield ''.join(parts)
                raw_stack.append(lower_tag)

            elif kind is END:
                tag = text_type(data)
                lower_tag = tag.lower()
                if raw_stack:
                    raw_stack.pop()
                if lower_tag not in self._EMPTY_ELEMS:
                    yield '</%s>' % tag

            elif kind is TEXT:
                if raw_stack and raw_stack[-1] in self._RAW_TEXT:
                    yield text_type(data)
                else:
                    yield _escape_text(data)

            elif kind is DOCTYPE:
                if not have_doctype:
                    yield _doctype_text(data)
                    have_doctype = True

            elif kind is COMMENT:
                yield '<!--%s-->' % text_type(data)

            elif kind is PI:
                target, text = data
                yield '<?%s%s?>' % (target, (' ' + text) if text else '')


class TextSerializer(object):
    def __init__(self, strip_whitespace=True):
        self.filters = []
        if strip_whitespace:
            self.filters.append(WhitespaceFilter())

    def __call__(self, stream):
        for filter_ in self.filters:
            stream = filter_(stream)
        cdata = False
        for kind, data, pos in stream:
            if kind is START_CDATA:
                cdata = True
            elif kind is END_CDATA:
                cdata = False
            elif kind is TEXT:
                yield text_type(data)