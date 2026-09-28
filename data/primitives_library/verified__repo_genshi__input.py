from itertools import chain
import codecs
import re
from xml.parsers import expat

from genshi.compat import html_entities, html_parser, text_type, unichr, \
                          StringIO, BytesIO
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
                        END_NS, START_CDATA, END_CDATA, PI, COMMENT


__all__ = ['ET', 'ParseError', 'XMLParser', 'XML', 'HTMLParser', 'HTML']
__docformat__ = 'restructuredtext en'


def ET(element):
    tag = QName(element.tag.lstrip('{'))
    attrs = Attrs((QName(name.lstrip('{')), value)
                  for name, value in element.items())

    yield START, (tag, attrs), (None, -1, -1)
    if element.text:
        yield TEXT, element.text, (None, -1, -1)

    for child in element:
        for event in ET(child):
            yield event

    yield END, tag, (None, -1, -1)

    if element.tail:
        yield TEXT, element.tail, (None, -1, -1)


class ParseError(Exception):

    def __init__(self, message, filename=None, lineno=-1, offset=-1):
        self.msg = message
        if filename:
            message = '%s, in %s' % (message, filename)
        Exception.__init__(self, message)
        self.filename = filename or '<string>'
        self.lineno = lineno
        self.offset = offset


def _coalesce(stream):
    pending = None

    for event in stream:
        kind, data, pos = event
        if kind is TEXT:
            if pending is None:
                pending = event
            else:
                pending = (TEXT, pending[1] + data, pending[2])
        else:
            if pending is not None:
                yield pending
                pending = None
            yield event

    if pending is not None:
        yield pending


class XMLParser(object):

    _entitydefs = [
        '<!ENTITY %s "&#%d;">' % (name, value)
        for name, value in html_entities.name2codepoint.items()
    ]
    _external_dtd = '\n'.join(_entitydefs).encode('utf-8')

    def __init__(self, source, filename=None, encoding=None):
        self.source = source
        self.filename = filename

        parser = expat.ParserCreate(encoding, '}')
        parser.buffer_text = True
        if hasattr(parser, 'returns_unicode'):
            parser.returns_unicode = True
        parser.ordered_attributes = True

        parser.StartElementHandler = self._handle_start
        parser.EndElementHandler = self._handle_end
        parser.CharacterDataHandler = self._handle_data
        parser.StartDoctypeDeclHandler = self._handle_doctype
        parser.StartNamespaceDeclHandler = self._handle_start_ns
        parser.EndNamespaceDeclHandler = self._handle_end_ns
        parser.StartCdataSectionHandler = self._handle_start_cdata
        parser.EndCdataSectionHandler = self._handle_end_cdata
        parser.ProcessingInstructionHandler = self._handle_pi
        parser.XmlDeclHandler = self._handle_xml_decl
        parser.CommentHandler = self._handle_comment
        parser.DefaultHandler = self._handle_other
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_ALWAYS)
        parser.UseForeignDTD()
        parser.ExternalEntityRefHandler = self._build_foreign

        self.expat = parser
        self._queue = []

    def parse(self):
        def generate():
            finished = False
            try:
                while True:
                    while not finished and not self._queue:
                        data = self.source.read(4 * 1024)
                        if not data:
                            if hasattr(self, 'expat'):
                                self.expat.Parse(b'', True)
                                del self.expat
                            finished = True
                        else:
                            if isinstance(data, text_type):
                                data = data.encode('utf-8')
                            self.expat.Parse(data, False)

                    for event in self._queue:
                        yield event
                    self._queue = []

                    if finished:
                        break
            except expat.ExpatError as error:
                raise ParseError(str(error), self.filename,
                                 error.lineno, error.offset)

        return Stream(generate()).filter(_coalesce)

    def __iter__(self):
        return iter(self.parse())

    def _build_foreign(self, context, base, sysid, pubid):
        parser = self.expat.ExternalEntityParserCreate(context)
        parser.ParseFile(BytesIO(self._external_dtd))
        return 1

    def _enqueue(self, kind, data=None, pos=None):
        if pos is None:
            pos = self._getpos()

        if kind is TEXT:
            if '\n' in data:
                line_count = len(data.splitlines())
                pos = (pos[0], pos[1] - line_count + 1, -1)
            else:
                pos = (pos[0], pos[1], pos[2] - len(data))

        self._queue.append((kind, data, pos))

    def _getpos_unknown(self):
        return self.filename, -1, -1

    def _getpos(self):
        return (self.filename, self.expat.CurrentLineNumber,
                self.expat.CurrentColumnNumber)

    def _handle_start(self, tag, attributes):
        pairs = zip(*[iter(attributes)] * 2)
        attrs = Attrs((QName(name), value) for name, value in pairs)
        self._enqueue(START, (QName(tag), attrs))

    def _handle_end(self, tag):
        self._enqueue(END, QName(tag))

    def _handle_data(self, text):
        self._enqueue(TEXT, text)

    def _handle_xml_decl(self, version, encoding, standalone):
        self._enqueue(XML_DECL, (version, encoding, standalone))

    def _handle_doctype(self, name, sysid, pubid, has_internal_subset):
        self._enqueue(DOCTYPE, (name, pubid, sysid))

    def _handle_start_ns(self, prefix, uri):
        self._enqueue(START_NS, (prefix or '', uri))

    def _handle_end_ns(self, prefix):
        self._enqueue(END_NS, prefix or '')

    def _handle_start_cdata(self):
        self._enqueue(START_CDATA)

    def _handle_end_cdata(self):
        self._enqueue(END_CDATA)

    def _handle_pi(self, target, data):
        self._enqueue(PI, (target, data))

    def _handle_comment(self, text):
        self._enqueue(COMMENT, text)

    def _handle_other(self, text):
        if text.startswith('&') and text.endswith(';'):
            name = text[1:-1]
            try:
                text = unichr(html_entities.name2codepoint[name])
            except KeyError:
                pass
            self._enqueue(TEXT, text)


def XML(text, encoding=None):
    if isinstance(text, text_type):
        source = StringIO(text)
    else:
        source = BytesIO(text)
    return XMLParser(source, encoding=encoding).parse()


class HTMLParser(html_parser.HTMLParser):

    def __init__(self, source, filename=None, encoding=None):
        try:
            html_parser.HTMLParser.__init__(self, convert_charrefs=False)
        except TypeError:
            html_parser.HTMLParser.__init__(self)

        self.source = source
        self.filename = filename
        self.encoding = encoding
        self._queue = []

    def parse(self):
        def generate():
            decoder = None
            if self.encoding:
                decoder = codecs.getincrementaldecoder(self.encoding)()
            else:
                decoder = codecs.getincrementaldecoder('utf-8')()

            try:
                while True:
                    data = self.source.read(4 * 1024)
                    if not data:
                        break

                    if not isinstance(data, text_type):
                        data = decoder.decode(data, False)

                    if data:
                        self.feed(data)

                    for event in self._queue:
                        yield event
                    self._queue = []

                tail = decoder.decode(b'', True)
                if tail:
                    self.feed(tail)

                self.close()

                for event in self._queue:
                    yield event
                self._queue = []
            except Exception as error:
                parse_error = getattr(html_parser, 'HTMLParseError', None)
                if parse_error is not None and isinstance(error, parse_error):
                    line, offset = self.getpos()
                    raise ParseError(str(error), self.filename, line, offset)
                raise

        return Stream(generate()).filter(_coalesce)

    def __iter__(self):
        return iter(self.parse())

    def _enqueue(self, kind, data=None, pos=None):
        if pos is None:
            line, offset = self.getpos()
            pos = self.filename, line, offset
        self._queue.append((kind, data, pos))

    def handle_starttag(self, tag, attrs):
        attributes = Attrs((QName(name), value) for name, value in attrs)
        self._enqueue(START, (QName(tag), attributes))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self._enqueue(END, QName(tag))

    def handle_data(self, data):
        self._enqueue(TEXT, data)

    def handle_entityref(self, name):
        try:
            data = unichr(html_entities.name2codepoint[name])
        except KeyError:
            data = '&%s;' % name
        self._enqueue(TEXT, data)

    def handle_charref(self, name):
        try:
            if name.lower().startswith('x'):
                data = unichr(int(name[1:], 16))
            else:
                data = unichr(int(name, 10))
        except (ValueError, OverflowError):
            data = '&#%s;' % name
        self._enqueue(TEXT, data)

    def handle_comment(self, data):
        self._enqueue(COMMENT, data)

    def handle_pi(self, data):
        data = data.rstrip('?').strip()
        if not data:
            return
        parts = data.split(None, 1)
        target = parts[0]
        content = parts[1] if len(parts) > 1 else ''
        self._enqueue(PI, (target, content))

    def handle_decl(self, decl):
        match = re.match(r'(?is)^\s*doctype\s+([^\s>]+)(.*)$', decl)
        if not match:
            return

        name = match.group(1)
        rest = match.group(2).strip()
        pubid = None
        sysid = None

        public = re.match(
            r'(?is)^public\s+([\'"])(.*?)\1(?:\s+([\'"])(.*?)\3)?', rest
        )
        system = re.match(r'(?is)^system\s+([\'"])(.*?)\1', rest)

        if public:
            pubid = public.group(2)
            sysid = public.group(4)
        elif system:
            sysid = system.group(2)

        self._enqueue(DOCTYPE, (name, pubid, sysid))


def HTML(text, encoding=None):
    if isinstance(text, text_type):
        source = StringIO(text)
    else:
        source = BytesIO(text)
    return HTMLParser(source, encoding=encoding).parse()