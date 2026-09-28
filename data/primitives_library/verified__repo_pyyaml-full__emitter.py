from .error import YAMLError
from .events import *

__all__ = ['Emitter', 'EmitterError']


class EmitterError(YAMLError):
    pass


class ScalarAnalysis:
    def __init__(self, scalar, empty, multiline, allow_flow_plain,
                 allow_block_plain, allow_single_quoted,
                 allow_double_quoted, allow_block):
        self.scalar = scalar
        self.empty = empty
        self.multiline = multiline
        self.allow_flow_plain = allow_flow_plain
        self.allow_block_plain = allow_block_plain
        self.allow_single_quoted = allow_single_quoted
        self.allow_double_quoted = allow_double_quoted
        self.allow_block = allow_block


class Emitter:

    DEFAULT_TAG_PREFIXES = {
        '!': '!',
        'tag:yaml.org,2002:': '!!',
    }

    ESCAPE_REPLACEMENTS = {
        '\0': '0',
        '\x07': 'a',
        '\x08': 'b',
        '\x09': 't',
        '\x0A': 'n',
        '\x0B': 'v',
        '\x0C': 'f',
        '\x0D': 'r',
        '\x1B': 'e',
        '"': '"',
        '\\': '\\',
        '\x85': 'N',
        '\xA0': '_',
        '\u2028': 'L',
        '\u2029': 'P',
    }

    def __init__(self, stream, canonical=None, indent=None, width=None,
                 allow_unicode=None, line_break=None):
        self.stream = stream
        self.encoding = None
        self.states = []
        self.state = self.expect_stream_start
        self.events = []
        self.event = None
        self.indents = []
        self.indent = None
        self.flow_level = 0

        self.root_context = False
        self.sequence_context = False
        self.mapping_context = False
        self.simple_key_context = False

        self.line = 0
        self.column = 0
        self.whitespace = True
        self.indention = True
        self.open_ended = False

        self.canonical = canonical
        self.allow_unicode = allow_unicode
        self.best_indent = 2
        if indent and 1 < indent < 10:
            self.best_indent = indent
        self.best_width = 80
        if width and width > self.best_indent * 2:
            self.best_width = width
        self.best_line_break = '\n'
        if line_break in ['\r', '\n', '\r\n']:
            self.best_line_break = line_break

        self.tag_prefixes = None
        self.prepared_anchor = None
        self.prepared_tag = None
        self.analysis = None
        self.style = None

    def dispose(self):
        self.states = []
        self.state = None

    def emit(self, event):
        self.events.append(event)
        while not self.need_more_events():
            self.event = self.events.pop(0)
            self.state()
            self.event = None

    def need_more_events(self):
        if not self.events:
            return True
        head = self.events[0]
        if isinstance(head, DocumentStartEvent):
            return self.need_events(1)
        if isinstance(head, SequenceStartEvent):
            return self.need_events(2)
        if isinstance(head, MappingStartEvent):
            return self.need_events(3)
        return False

    def need_events(self, count):
        nesting = 0
        for event in self.events[1:]:
            if isinstance(event, (DocumentStartEvent, CollectionStartEvent)):
                nesting += 1
            elif isinstance(event, (DocumentEndEvent, CollectionEndEvent)):
                nesting -= 1
            elif isinstance(event, StreamEndEvent):
                nesting = -1
            if nesting < 0:
                return False
        return len(self.events) < count + 1

    def increase_indent(self, flow=False, indentless=False):
        self.indents.append(self.indent)
        if self.indent is None:
            self.indent = self.best_indent if flow else 0
        elif not indentless:
            self.indent += self.best_indent

    def expect_stream_start(self):
        if not isinstance(self.event, StreamStartEvent):
            raise EmitterError("expected StreamStartEvent, but got %s" % self.event)
        if self.event.encoding and not hasattr(self.stream, 'encoding'):
            self.encoding = self.event.encoding
        self.write_stream_start()
        self.state = self.expect_first_document_start

    def expect_nothing(self):
        raise EmitterError("expected nothing, but got %s" % self.event)

    def expect_first_document_start(self):
        self.expect_document_start(first=True)

    def expect_document_start(self, first=False):
        if isinstance(self.event, DocumentStartEvent):
            if (self.event.version or self.event.tags) and self.open_ended:
                self.write_indicator('...', True)
                self.write_indent()
            if self.event.version:
                self.write_version_directive(self.prepare_version(self.event.version))
            self.tag_prefixes = self.DEFAULT_TAG_PREFIXES.copy()
            if self.event.tags:
                for handle in sorted(self.event.tags):
                    prefix = self.event.tags[handle]
                    self.tag_prefixes[prefix] = handle
                    self.write_tag_directive(
                        self.prepare_tag_handle(handle),
                        self.prepare_tag_prefix(prefix))
            implicit = (first and not self.event.explicit and not self.canonical
                        and not self.event.version and not self.event.tags
                        and not self.check_empty_document())
            if not implicit:
                self.write_indent()
                self.write_indicator('---', True)
                if self.canonical:
                    self.write_indent()
            self.state = self.expect_document_root
        elif isinstance(self.event, StreamEndEvent):
            if self.open_ended:
                self.write_indicator('...', True)
                self.write_indent()
            self.write_stream_end()
            self.state = self.expect_nothing
        else:
            raise EmitterError("expected DocumentStartEvent, but got %s" % self.event)

    def expect_document_end(self):
        if not isinstance(self.event, DocumentEndEvent):
            raise EmitterError("expected DocumentEndEvent, but got %s" % self.event)
        self.write_indent()
        if self.event.explicit:
            self.write_indicator('...', True)
            self.write_indent()
        self.flush_stream()
        self.state = self.expect_document_start

    def expect_document_root(self):
        self.states.append(self.expect_document_end)
        self.expect_node(root=True)

    def expect_node(self, root=False, sequence=False, mapping=False,
                    simple_key=False):
        self.root_context = root
        self.sequence_context = sequence
        self.mapping_context = mapping
        self.simple_key_context = simple_key

        if isinstance(self.event, AliasEvent):
            self.expect_alias()
        elif isinstance(self.event, (ScalarEvent, CollectionStartEvent)):
            self.process_anchor('&')
            self.process_tag()
            if isinstance(self.event, ScalarEvent):
                self.expect_scalar()
            elif isinstance(self.event, SequenceStartEvent):
                if (self.flow_level or self.canonical or self.event.flow_style
                        or self.check_empty_sequence()):
                    self.expect_flow_sequence()
                else:
                    self.expect_block_sequence()
            elif isinstance(self.event, MappingStartEvent):
                if (self.flow_level or self.canonical or self.event.flow_style
                        or self.check_empty_mapping()):
                    self.expect_flow_mapping()
                else:
                    self.expect_block_mapping()
        else:
            raise EmitterError("expected NodeEvent, but got %s" % self.event)

    def expect_alias(self):
        if self.event.anchor is None:
            raise EmitterError("anchor is not specified for alias")
        self.process_anchor('*')
        self.state = self.states.pop()

    def expect_scalar(self):
        self.increase_indent(flow=True)
        self.process_scalar()
        self.indent = self.indents.pop()
        self.state = self.states.pop()

    def expect_flow_sequence(self):
        self.write_indicator('[', True, whitespace=True)
        self.flow_level += 1
        self.increase_indent(flow=True)
        self.state = self.expect_first_flow_sequence_item

    def expect_first_flow_sequence_item(self):
        if isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            self.write_indicator(']', False)
            self.state = self.states.pop()
        else:
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            self.states.append(self.expect_flow_sequence_item)
            self.expect_node(sequence=True)

    def expect_flow_sequence_item(self):
        if isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            if self.canonical:
                self.write_indicator(',', False)
                self.write_indent()
            self.write_indicator(']', False)
            self.state = self.states.pop()
        else:
            self.write_indicator(',', False)
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            self.states.append(self.expect_flow_sequence_item)
            self.expect_node(sequence=True)

    def expect_flow_mapping(self):
        self.write_indicator('{', True, whitespace=True)
        self.flow_level += 1
        self.increase_indent(flow=True)
        self.state = self.expect_first_flow_mapping_key

    def expect_first_flow_mapping_key(self):
        if isinstance(self.event, MappingEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            self.write_indicator('}', False)
            self.state = self.states.pop()
        else:
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            if not self.canonical and self.check_simple_key():
                self.states.append(self.expect_flow_mapping_simple_value)
                self.expect_node(mapping=True, simple_key=True)
            else:
                self.write_indicator('?', True)
                self.states.append(self.expect_flow_mapping_value)
                self.expect_node(mapping=True)

    def expect_flow_mapping_key(self):
        if isinstance(self.event, MappingEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            if self.canonical:
                self.write_indicator(',', False)
                self.write_indent()
            self.write_indicator('}', False)
            self.state = self.states.pop()
        else:
            self.write_indicator(',', False)
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            if not self.canonical and self.check_simple_key():
                self.states.append(self.expect_flow_mapping_simple_value)
                self.expect_node(mapping=True, simple_key=True)
            else:
                self.write_indicator('?', True)
                self.states.append(self.expect_flow_mapping_value)
                self.expect_node(mapping=True)

    def expect_flow_mapping_simple_value(self):
        self.write_indicator(':', False)
        self.states.append(self.expect_flow_mapping_key)
        self.expect_node(mapping=True)

    def expect_flow_mapping_value(self):
        if isinstance(self.event, MappingEndEvent):
            self.write_indicator(':', False)
            self.state = self.expect_flow_mapping_key
        else:
            self.write_indicator(':', True)
            self.states.append(self.expect_flow_mapping_key)
            self.expect_node(mapping=True)

    def expect_block_sequence(self):
        indentless = (self.mapping_context and not self.indention)
        self.increase_indent(flow=False, indentless=indentless)
        self.state = self.expect_first_block_sequence_item

    def expect_first_block_sequence_item(self):
        return self.expect_block_sequence_item(first=True)

    def expect_block_sequence_item(self, first=False):
        if isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        else:
            self.write_indent()
            self.write_indicator('-', True, indention=True)
            self.states.append(self.expect_block_sequence_item)
            self.expect_node(sequence=True)

    def expect_block_mapping(self):
        self.increase_indent(flow=False)
        self.state = self.expect_first_block_mapping_key

    def expect_first_block_mapping_key(self):
        self.expect_block_mapping_key(first=True)

    def expect_block_mapping_key(self, first=False):
        if isinstance(self.event, MappingEndEvent):
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        else:
            self.write_indent()
            if self.check_simple_key():
                self.states.append(self.expect_block_mapping_simple_value)
                self.expect_node(mapping=True, simple_key=True)
            else:
                self.write_indicator('?', True, indention=True)
                self.states.append(self.expect_block_mapping_value)
                self.expect_node(mapping=True)

    def expect_block_mapping_simple_value(self):
        self.write_indicator(':', False, indention=True)
        self.states.append(self.expect_block_mapping_key)
        self.expect_node(mapping=True)

    def expect_block_mapping_value(self):
        if isinstance(self.event, MappingEndEvent):
            self.write_indicator(':', False, indention=True)
            self.state = self.expect_block_mapping_key
        else:
            self.write_indent()
            self.write_indicator(':', True, indention=True)
            self.states.append(self.expect_block_mapping_key)
            self.expect_node(mapping=True)

    def check_empty_document(self):
        return (isinstance(self.event, DocumentStartEvent)
                and isinstance(self.events[0], ScalarEvent)
                and self.events[0].anchor is None
                and self.events[0].tag is None
                and self.events[0].implicit
                and self.events[0].value == '')

    def check_empty_sequence(self):
        return (isinstance(self.event, SequenceStartEvent)
                and isinstance(self.events[0], SequenceEndEvent))

    def check_empty_mapping(self):
        return (isinstance(self.event, MappingStartEvent)
                and isinstance(self.events[0], MappingEndEvent))

    def check_simple_key(self):
        length = 0
        if isinstance(self.event, NodeEvent) and self.event.anchor is not None:
            if self.prepared_anchor is None:
                self.prepared_anchor = self.prepare_anchor(self.event.anchor)
            length += len(self.prepared_anchor)
        if isinstance(self.event, ScalarEvent):
            if self.analysis is None:
                self.analysis = self.analyze_scalar(self.event.value)
            length += len(self.analysis.scalar)
        if isinstance(self.event, NodeEvent) and self.event.tag is not None:
            if self.prepared_tag is None:
                self.prepared_tag = self.prepare_tag(self.event.tag)
            length += len(self.prepared_tag)
        return length < 128 and (isinstance(self.event, AliasEvent)
                                 or (isinstance(self.event, ScalarEvent)
                                     and not self.analysis.empty
                                     and not self.analysis.multiline))

    def process_anchor(self, indicator):
        if self.event.anchor is None:
            self.prepared_anchor = None
            return
        if self.prepared_anchor is None:
            self.prepared_anchor = self.prepare_anchor(self.event.anchor)
        self.write_indicator(indicator + self.prepared_anchor, True)

    def process_tag(self):
        tag = self.event.tag
        if isinstance(self.event, ScalarEvent):
            if self.style is None:
                self.style = self.choose_scalar_style()
            if ((not self.canonical or tag is None)
                    and (self.event.implicit[0] if self.style == '' else self.event.implicit[1])):
                self.prepared_tag = None
                return
        elif tag is None:
            self.prepared_tag = None
            return

        if self.prepared_tag is None:
            self.prepared_tag = self.prepare_tag(tag)
        self.write_indicator(self.prepared_tag, True)

    def choose_scalar_style(self):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)

        if self.event.style == '"' or self.canonical:
            return '"'

        if not self.event.style and self.event.implicit[0]:
            if (not (self.simple_key_context and
                     (self.analysis.empty or self.analysis.multiline))
                    and (self.flow_level and self.analysis.allow_flow_plain
                         or not self.flow_level and self.analysis.allow_block_plain)):
                return ''

        if self.event.style and self.event.style in '|>':
            if (not self.flow_level and not self.simple_key_context
                    and self.analysis.allow_block):
                return self.event.style

        if not self.event.style or self.event.style == "'":
            if (self.analysis.allow_single_quoted
                    and not (self.simple_key_context and self.analysis.multiline)):
                return "'"

        if self.analysis.allow_double_quoted:
            return '"'

        return '"'

    def process_scalar(self):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)
        if self.style is None:
            self.style = self.choose_scalar_style()

        split = not self.simple_key_context
        if self.style == '':
            self.write_plain(self.analysis.scalar, split)
        elif self.style == "'":
            self.write_single_quoted(self.analysis.scalar, split)
        elif self.style == '"':
            self.write_double_quoted(self.analysis.scalar, split)
        elif self.style == '>':
            self.write_folded(self.analysis.scalar)
        elif self.style == '|':
            self.write_literal(self.analysis.scalar)

        self.analysis = None
        self.style = None

    def prepare_version(self, version):
        major, minor = version
        if major != 1:
            raise EmitterError("unsupported YAML version: %d.%d" % (major, minor))
        return '%d.%d' % (major, minor)

    def prepare_tag_handle(self, handle):
        if not handle:
            raise EmitterError("tag handle must not be empty")
        if handle[0] != '!' or handle[-1] != '!':
            raise EmitterError("tag handle must start and end with '!'")
        if handle == '!':
            return handle
        return '!' + ''.join(
            ch if ('0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'
                   or ch in '-_') else self._escape_uri(ch)
            for ch in handle[1:-1]) + '!'

    def prepare_tag_prefix(self, prefix):
        if not prefix:
            raise EmitterError("tag prefix must not be empty")
        return ''.join(
            ch if ('0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'
                   or ch in '-;/?:@&=+$,_.!~*\'()[]%#')
            else self._escape_uri(ch)
            for ch in prefix)

    def prepare_tag(self, tag):
        if not tag:
            raise EmitterError("tag must not be empty")
        prefixes = sorted(self.tag_prefixes, key=len, reverse=True)
        for prefix in prefixes:
            if tag.startswith(prefix):
                handle = self.tag_prefixes[prefix]
                suffix = tag[len(prefix):]
                return handle + self.prepare_tag_prefix(suffix)
        return '!<' + self.prepare_tag_prefix(tag) + '>'

    def prepare_anchor(self, anchor):
        if not anchor:
            raise EmitterError("anchor must not be empty")
        result = []
        for ch in anchor:
            if ('0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'
                    or ch in '-_'):
                result.append(ch)
            else:
                raise EmitterError("invalid character %r in the anchor: %r"
                                   % (ch, anchor))
        return ''.join(result)

    def analyze_scalar(self, scalar):
        if not scalar:
            return ScalarAnalysis(scalar, True, False, False, True,
                                  True, True, False)

        block_indicators = scalar.startswith('---') or scalar.startswith('...')
        flow_indicators = scalar.startswith(',?[]{}')
        if scalar.startswith(':'):
            flow_indicators = True
            if len(scalar) == 1 or scalar[1].isspace():
                block_indicators = True
        if scalar.startswith('-') and (len(scalar) == 1 or scalar[1].isspace()):
            block_indicators = True

        line_breaks = False
        special_characters = False
        leading_space = False
        leading_break = False
        trailing_space = False
        trailing_break = False
        break_space = False
        space_break = False
        flow_space = False
        block_space = False

        preceded_by_whitespace = True
        preceded_by_break = False
        index = 0
        while index < len(scalar):
            ch = scalar[index]
            followed_by_whitespace = (index + 1 == len(scalar)
                                      or scalar[index + 1] in '\0 \t\r\n\x85\u2028\u2029')
            if index == 0:
                if ch in '#,[]{}&*!|>\'\"%@`':
                    flow_indicators = True
                    block_indicators = True
                if ch in '?:':
                    flow_indicators = True
                    if followed_by_whitespace:
                        block_indicators = True
                if ch == '-' and followed_by_whitespace:
                    flow_indicators = True
                    block_indicators = True
            else:
                if ch in ',?[]{}':
                    flow_indicators = True
                if ch == ':':
                    flow_indicators = True
                    if followed_by_whitespace:
                        block_indicators = True
                if ch == '#' and preceded_by_whitespace:
                    flow_indicators = True
                    block_indicators = True

            if ch in '\0\x07\x08\x09\x0B\x0C\x1B':
                special_characters = True
            elif ('\x00' <= ch <= '\x08' or '\x0B' <= ch <= '\x1F'
                  or '\x7F' <= ch <= '\x84' or '\x86' <= ch <= '\x9F'
                  or (not self.allow_unicode and
                      ('\uD800' <= ch <= '\uDFFF' or ch > '\uFFFD'))):
                special_characters = True

            if ch == ' ':
                if index == 0:
                    leading_space = True
                if index + 1 == len(scalar):
                    trailing_space = True
                if preceded_by_break:
                    break_space = True
            elif ch in '\n\x85\u2028\u2029':
                line_breaks = True
                if index == 0:
                    leading_break = True
                if index + 1 == len(scalar):
                    trailing_break = True
                if preceded_by_whitespace:
                    space_break = True

            if ch == ' ' and (self.flow_level or True) and preceded_by_whitespace:
                flow_space = flow_space or False
            if ch == ' ' and preceded_by_whitespace:
                block_space = block_space or False

            preceded_by_whitespace = ch in '\0 \t\r\n\x85\u2028\u2029'
            preceded_by_break = ch in '\n\x85\u2028\u2029'
            index += 1

        allow_flow_plain = True
        allow_block_plain = True
        allow_single_quoted = True
        allow_double_quoted = True
        allow_block = True

        if leading_space or leading_break or trailing_space or trailing_break:
            allow_flow_plain = allow_block_plain = False
        if trailing_space:
            allow_block = False
        if break_space:
            allow_flow_plain = allow_block_plain = allow_single_quoted = False
        if space_break or special_characters:
            allow_flow_plain = allow_block_plain = allow_single_quoted = allow_block = False
        if line_breaks:
            allow_flow_plain = allow_block_plain = False
        if flow_indicators:
            allow_flow_plain = False
        if block_indicators:
            allow_block_plain = False

        return ScalarAnalysis(scalar, False, line_breaks,
                              allow_flow_plain, allow_block_plain,
                              allow_single_quoted, allow_double_quoted,
                              allow_block)

    def _escape_uri(self, char):
        data = char.encode('utf-8')
        return ''.join('%%%02X' % byte for byte in data)

    def flush_stream(self):
        if hasattr(self.stream, 'flush'):
            self.stream.flush()

    def write_stream_start(self):
        pass

    def write_stream_end(self):
        self.flush_stream()

    def write_indicator(self, indicator, need_whitespace,
                        whitespace=False, indention=False):
        if self.whitespace or not need_whitespace:
            pass
        else:
            self.write_plain(' ', False)
        self.whitespace = whitespace
        self.indention = self.indention and indention
        self.column += len(indicator)
        self.stream.write(indicator)

    def write_indent(self):
        indent = self.indent or 0
        if not self.indention or self.column > indent:
            self.write_line_break()
        while self.column < indent:
            self.stream.write(' ')
            self.column += 1
        self.whitespace = True
        self.indention = True

    def write_line_break(self, data=None):
        if data is None:
            data = self.best_line_break
        self.stream.write(data)
        self.line += 1
        self.column = 0
        self.whitespace = True
        self.indention = True

    def write_version_directive(self, version_text):
        self.write_indicator('%YAML %s' % version_text, True)
        self.write_indent()

    def write_tag_directive(self, handle_text, prefix_text):
        self.write_indicator('%TAG %s %s' % (handle_text, prefix_text), True)
        self.write_indent()

    def determine_block_hints(self, text):
        hints = ''
        if text and text[0] in ' \n\x85\u2028\u2029':
            hints += str(self.best_indent)
        if text and text[-1] not in '\n\x85\u2028\u2029':
            hints += '-'
        elif len(text) > 1 and text[-2] in '\n\x85\u2028\u2029':
            hints += '+'
        return hints

    def write_folded(self, text):
        hints = self.determine_block_hints(text)
        self.write_indicator('>' + hints, True)
        if hints[-1:] == '+':
            self.open_ended = True
        self.write_line_break()
        leading_space = True
        spaces = False
        breaks = True
        start = end = 0
        while end <= len(text):
            ch = text[end] if end < len(text) else None
            if breaks:
                if ch is None or ch not in '\n\x85\u2028\u2029':
                    if not leading_space and ch is not None and ch != ' ' and text[start] == '\n':
                        self.write_line_break()
                    leading_space = ch == ' '
                    for c in text[start:end]:
                        self.write_line_break(c)
                    if ch is not None:
                        self.write_indent()
                    breaks = False
                    start = end
            elif spaces:
                if ch != ' ':
                    if start + 1 == end and self.column > self.best_width:
                        self.write_indent()
                    else:
                        data = text[start:end]
                        self.stream.write(data)
                        self.column += len(data)
                    start = end
                    spaces = False
            else:
                if ch is None or ch in ' \n\x85\u2028\u2029':
                    data = text[start:end]
                    self.stream.write(data)
                    self.column += len(data)
                    if ch is None:
                        self.write_line_break()
                    start = end
                    spaces = ch == ' '
                    breaks = ch in '\n\x85\u2028\u2029'
            end += 1

    def write_literal(self, text):
        hints = self.determine_block_hints(text)
        self.write_indicator('|' + hints, True)
        if hints[-1:] == '+':
            self.open_ended = True
        self.write_line_break()
        breaks = True
        start = end = 0
        while end <= len(text):
            ch = text[end] if end < len(text) else None
            if breaks:
                if ch is None or ch not in '\n\x85\u2028\u2029':
                    for c in text[start:end]:
                        self.write_line_break(c)
                    if ch is not None:
                        self.write_indent()
                    breaks = False
                    start = end
            else:
                if ch is None or ch in '\n\x85\u2028\u2029':
                    data = text[start:end]
                    self.stream.write(data)
                    self.column += len(data)
                    if ch is None:
                        self.write_line_break()
                    breaks = True
                    start = end
            end += 1

    def write_plain(self, text, split=True):
        if not text:
            return
        if not self.whitespace:
            self.stream.write(' ')
            self.column += 1
        self.whitespace = False
        self.indention = False
        spaces = False
        breaks = False
        start = end = 0
        while end <= len(text):
            ch = text[end] if end < len(text) else None
            if spaces:
                if ch != ' ':
                    if start + 1 == end and self.column > self.best_width and split:
                        self.write_indent()
                    else:
                        data = text[start:end]
                        self.stream.write(data)
                        self.column += len(data)
                    start = end
                    spaces = False
            elif breaks:
                if ch not in '\n\x85\u2028\u2029':
                    if text[start] == '\n':
                        self.write_line_break()
                    for c in text[start:end]:
                        if c == '\n':
                            self.write_line_break()
                        else:
                            self.write_line_break(c)
                    self.write_indent()
                    start = end
                    breaks = False
            else:
                if ch is None or ch in ' \n\x85\u2028\u2029':
                    data = text[start:end]
                    self.stream.write(data)
                    self.column += len(data)
                    start = end
                    spaces = ch == ' '
                    breaks = ch in '\n\x85\u2028\u2029'
            end += 1

    def write_single_quoted(self, text, split=True):
        self.write_indicator("'", True)
        spaces = False
        breaks = False
        start = end = 0
        while end <= len(text):
            ch = text[end] if end < len(text) else None
            if spaces:
                if ch != ' ':
                    if start + 1 == end and self.column > self.best_width and split:
                        self.write_indent()
                    else:
                        data = text[start:end]
                        self.stream.write(data)
                        self.column += len(data)
                    start = end
                    spaces = False
            elif breaks:
                if ch not in '\n\x85\u2028\u2029':
                    if text[start] == '\n':
                        self.write_line_break()
                    for c in text[start:end]:
                        self.write_line_break(c)
                    self.write_indent()
                    start = end
                    breaks = False
            else:
                if ch is None or ch in ' \n\x85\u2028\u2029\'':
                    data = text[start:end]
                    self.stream.write(data)
                    self.column += len(data)
                    if ch == "'":
                        self.stream.write("'")
                        self.column += 1
                    start = end
                    spaces = ch == ' '
                    breaks = ch in '\n\x85\u2028\u2029'
            end += 1
        self.write_indicator("'", False)

    def write_double_quoted(self, text, split=True):
        self.write_indicator('"', True)
        start = end = 0
        while end <= len(text):
            ch = text[end] if end < len(text) else None
            if ch is None or ch in '"\\\x00\x07\x08\x09\x0A\x0B\x0C\x0D\x1B\x85\xA0\u2028\u2029' or (
                    '\x00' <= ch <= '\x08' or '\x0B' <= ch <= '\x1F'
                    or '\x7F' <= ch <= '\x84' or '\x86' <= ch <= '\x9F'
                    or (not self.allow_unicode and
                        ('\uD800' <= ch <= '\uDFFF' or ch > '\uFFFD'))):
                if start < end:
                    data = text[start:end]
                    self.stream.write(data)
                    self.column += len(data)
                if ch is not None:
                    if ch in self.ESCAPE_REPLACEMENTS:
                        data = '\\' + self.ESCAPE_REPLACEMENTS[ch]
                    elif ch <= '\xFF':
                        data = '\\x%02X' % ord(ch)
                    elif ch <= '\uFFFF':
                        data = '\\u%04X' % ord(ch)
                    else:
                        data = '\\U%08X' % ord(ch)
                    self.stream.write(data)
                    self.column += len(data)
                start = end + 1
            if 0 < end < len(text) - 1 and text[end] == ' ' and self.column > self.best_width and split:
                self.write_indent()
                if start == end + 1:
                    self.stream.write('\\')
                    self.column += 1
            end += 1
        self.write_indicator('"', False)