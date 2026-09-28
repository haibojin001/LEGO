from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *

__all__ = ['Parser', 'ParserError']


class ParserError(MarkedYAMLError):
    pass


class Parser:

    DEFAULT_TAGS = {
        '!': '!',
        '!!': 'tag:yaml.org,2002:',
    }

    def __init__(self):
        self.current_event = None
        self.yaml_version = None
        self.tag_handles = {}
        self.states = []
        self.marks = []
        self.state = self.parse_stream_start

    def dispose(self):
        self.states = []
        self.state = None

    def check_event(self, *choices):
        if self.current_event is None and self.state is not None:
            self.current_event = self.state()
        if self.current_event is None:
            return False
        if not choices:
            return True
        return any(isinstance(self.current_event, choice) for choice in choices)

    def peek_event(self):
        if self.current_event is None and self.state is not None:
            self.current_event = self.state()
        return self.current_event

    def get_event(self):
        if self.current_event is None and self.state is not None:
            self.current_event = self.state()
        event = self.current_event
        self.current_event = None
        return event

    def parse_stream_start(self):
        token = self.get_token()
        self.state = self.parse_implicit_document_start
        return StreamStartEvent(token.start_mark, token.end_mark,
                                encoding=token.encoding)

    def parse_implicit_document_start(self):
        if not self.check_token(DirectiveToken, DocumentStartToken,
                                StreamEndToken):
            self.tag_handles = self.DEFAULT_TAGS
            token = self.peek_token()
            self.states.append(self.parse_document_end)
            self.state = self.parse_block_node
            return DocumentStartEvent(token.start_mark, token.start_mark,
                                      explicit=False)
        return self.parse_document_start()

    def parse_document_start(self):
        while self.check_token(DocumentEndToken):
            self.get_token()

        if self.check_token(StreamEndToken):
            token = self.get_token()
            if self.states or self.marks:
                raise AssertionError("parser state stack is not empty")
            self.state = None
            return StreamEndEvent(token.start_mark, token.end_mark)

        token = self.peek_token()
        start_mark = token.start_mark
        version, tags = self.process_directives()

        if not self.check_token(DocumentStartToken):
            token = self.peek_token()
            raise ParserError(
                None, None,
                "expected '<document start>', but found %r" % token.id,
                token.start_mark)

        token = self.get_token()
        self.states.append(self.parse_document_end)
        self.state = self.parse_document_content
        return DocumentStartEvent(start_mark, token.end_mark, explicit=True,
                                  version=version, tags=tags)

    def parse_document_end(self):
        token = self.peek_token()
        start_mark = end_mark = token.start_mark
        explicit = False
        if self.check_token(DocumentEndToken):
            token = self.get_token()
            end_mark = token.end_mark
            explicit = True
        self.state = self.parse_document_start
        return DocumentEndEvent(start_mark, end_mark, explicit=explicit)

    def parse_document_content(self):
        if self.check_token(DirectiveToken, DocumentStartToken,
                            DocumentEndToken, StreamEndToken):
            event = self.process_empty_scalar(self.peek_token().start_mark)
            self.state = self.states.pop()
            return event
        return self.parse_block_node()

    def process_directives(self):
        self.yaml_version = None
        self.tag_handles = {}
        tags = []

        while self.check_token(DirectiveToken):
            token = self.get_token()
            if token.name == 'YAML':
                if self.yaml_version is not None:
                    raise ParserError(None, None,
                                      "found duplicate YAML directive",
                                      token.start_mark)
                major, minor = token.value
                if major != 1:
                    raise ParserError(
                        None, None,
                        "found incompatible YAML document "
                        "(version 1.* is required)",
                        token.start_mark)
                self.yaml_version = token.value
            elif token.name == 'TAG':
                handle, prefix = token.value
                if handle in self.tag_handles:
                    raise ParserError(
                        None, None,
                        "duplicate tag handle %r" % handle,
                        token.start_mark)
                self.tag_handles[handle] = prefix
                tags.append(token.value)

        for handle, prefix in self.DEFAULT_TAGS.items():
            if handle not in self.tag_handles:
                self.tag_handles[handle] = prefix

        return self.yaml_version, tags

    def process_empty_scalar(self, mark):
        return ScalarEvent(None, None, (True, False), '', mark, mark)

    def _block_node_start(self):
        return self.check_token(
            AliasToken, AnchorToken, TagToken, ScalarToken,
            BlockSequenceStartToken, BlockMappingStartToken,
            FlowSequenceStartToken, FlowMappingStartToken)

    def _block_node_or_indentless_start(self):
        return self.check_token(
            AliasToken, AnchorToken, TagToken, ScalarToken,
            BlockSequenceStartToken, BlockMappingStartToken,
            FlowSequenceStartToken, FlowMappingStartToken,
            BlockEntryToken)

    def _flow_node_start(self):
        return self.check_token(
            AliasToken, AnchorToken, TagToken, ScalarToken,
            FlowSequenceStartToken, FlowMappingStartToken)

    def parse_block_node(self):
        return self.parse_node(block=True, indentless_sequence=False)

    def parse_block_node_or_indentless_sequence(self):
        return self.parse_node(block=True, indentless_sequence=True)

    def parse_flow_node(self):
        return self.parse_node(block=False, indentless_sequence=False)

    def parse_node(self, block, indentless_sequence):
        if self.check_token(AliasToken):
            token = self.get_token()
            self.state = self.states.pop()
            return AliasEvent(token.value, token.start_mark, token.end_mark)

        anchor = None
        tag = None
        start_mark = end_mark = self.peek_token().start_mark

        if self.check_token(AnchorToken):
            token = self.get_token()
            start_mark = token.start_mark
            end_mark = token.end_mark
            anchor = token.value
            if self.check_token(TagToken):
                token = self.get_token()
                end_mark = token.end_mark
                tag = token.value
        elif self.check_token(TagToken):
            token = self.get_token()
            start_mark = token.start_mark
            end_mark = token.end_mark
            tag = token.value
            if self.check_token(AnchorToken):
                token = self.get_token()
                end_mark = token.end_mark
                anchor = token.value

        if tag is not None:
            handle, suffix = tag
            if handle is not None:
                if handle not in self.tag_handles:
                    raise ParserError(
                        "while parsing a node", start_mark,
                        "found undefined tag handle %r" % handle,
                        tag_token_mark(tag, self.peek_token(), start_mark))
                tag = self.tag_handles[handle] + suffix
            else:
                tag = suffix

        if indentless_sequence and self.check_token(BlockEntryToken):
            end_mark = self.peek_token().end_mark
            self.state = self.parse_indentless_sequence_entry
            return SequenceStartEvent(anchor, tag, tag is None,
                                      start_mark, end_mark,
                                      flow_style=False)

        if self.check_token(ScalarToken):
            token = self.get_token()
            end_mark = token.end_mark
            self.state = self.states.pop()
            if tag is None:
                implicit = (True, False)
            else:
                implicit = (False, False)
            return ScalarEvent(anchor, tag, implicit, token.value,
                               start_mark, end_mark, style=token.style)

        if self.check_token(FlowSequenceStartToken):
            end_mark = self.peek_token().end_mark
            self.state = self.parse_flow_sequence_first_entry
            return SequenceStartEvent(anchor, tag, tag is None,
                                      start_mark, end_mark,
                                      flow_style=True)

        if self.check_token(FlowMappingStartToken):
            end_mark = self.peek_token().end_mark
            self.state = self.parse_flow_mapping_first_key
            return MappingStartEvent(anchor, tag, tag is None,
                                     start_mark, end_mark,
                                     flow_style=True)

        if block and self.check_token(BlockSequenceStartToken):
            end_mark = self.peek_token().end_mark
            self.state = self.parse_block_sequence_first_entry
            return SequenceStartEvent(anchor, tag, tag is None,
                                      start_mark, end_mark,
                                      flow_style=False)

        if block and self.check_token(BlockMappingStartToken):
            end_mark = self.peek_token().end_mark
            self.state = self.parse_block_mapping_first_key
            return MappingStartEvent(anchor, tag, tag is None,
                                     start_mark, end_mark,
                                     flow_style=False)

        if anchor is not None or tag is not None:
            self.state = self.states.pop()
            return ScalarEvent(anchor, tag, (tag is None, False), '',
                               start_mark, end_mark)

        token = self.peek_token()
        context = "while parsing a %s node" % ("block" if block else "flow")
        raise ParserError(context, start_mark,
                          "expected the node content, but found %r" % token.id,
                          token.start_mark)

    def parse_block_sequence_first_entry(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        self.state = self.parse_block_sequence_entry
        return self.parse_block_sequence_entry()

    def parse_block_sequence_entry(self):
        if self.check_token(BlockEntryToken):
            token = self.get_token()
            if not self.check_token(BlockEntryToken, BlockEndToken):
                self.states.append(self.parse_block_sequence_entry)
                return self.parse_block_node()
            self.state = self.parse_block_sequence_entry
            return self.process_empty_scalar(token.end_mark)

        if self.check_token(BlockEndToken):
            token = self.get_token()
            self.state = self.states.pop()
            start_mark = self.marks.pop()
            return SequenceEndEvent(start_mark, token.end_mark)

        token = self.peek_token()
        raise ParserError("while parsing a block collection", self.marks[-1],
                          "expected <block end>, but found %r" % token.id,
                          token.start_mark)

    def parse_indentless_sequence_entry(self):
        if self.check_token(BlockEntryToken):
            token = self.get_token()
            if not self.check_token(BlockEntryToken, KeyToken,
                                    ValueToken, BlockEndToken,
                                    DocumentEndToken, StreamEndToken):
                self.states.append(self.parse_indentless_sequence_entry)
                return self.parse_block_node()
            self.state = self.parse_indentless_sequence_entry
            return self.process_empty_scalar(token.end_mark)

        token = self.peek_token()
        self.state = self.states.pop()
        return SequenceEndEvent(token.start_mark, token.start_mark)

    def parse_block_mapping_first_key(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        self.state = self.parse_block_mapping_key
        return self.parse_block_mapping_key()

    def parse_block_mapping_key(self):
        if self.check_token(KeyToken):
            token = self.get_token()
            if self._block_node_or_indentless_start():
                self.states.append(self.parse_block_mapping_value)
                return self.parse_block_node_or_indentless_sequence()
            self.state = self.parse_block_mapping_value
            return self.process_empty_scalar(token.end_mark)

        if self.check_token(ValueToken):
            self.state = self.parse_block_mapping_value
            return self.process_empty_scalar(self.peek_token().start_mark)

        if self.check_token(BlockEndToken):
            token = self.get_token()
            self.state = self.states.pop()
            return MappingEndEvent(self.marks.pop(), token.end_mark)

        token = self.peek_token()
        raise ParserError("while parsing a block mapping", self.marks[-1],
                          "expected <block end>, but found %r" % token.id,
                          token.start_mark)

    def parse_block_mapping_value(self):
        if self.check_token(ValueToken):
            token = self.get_token()
            if self._block_node_or_indentless_start():
                self.states.append(self.parse_block_mapping_key)
                return self.parse_block_node_or_indentless_sequence()
            self.state = self.parse_block_mapping_key
            return self.process_empty_scalar(token.end_mark)

        self.state = self.parse_block_mapping_key
        return self.process_empty_scalar(self.peek_token().start_mark)

    def parse_flow_sequence_first_entry(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        self.state = self.parse_flow_sequence_entry
        return self.parse_flow_sequence_entry(first=True)

    def parse_flow_sequence_entry(self, first=False):
        if self.check_token(FlowSequenceEndToken):
            token = self.get_token()
            self.state = self.states.pop()
            return SequenceEndEvent(self.marks.pop(), token.end_mark)

        if not first:
            if self.check_token(FlowEntryToken):
                self.get_token()
            else:
                token = self.peek_token()
                raise ParserError(
                    "while parsing a flow sequence", self.marks[-1],
                    "expected ',' or ']', but got %r" % token.id,
                    token.start_mark)

            if self.check_token(FlowSequenceEndToken):
                token = self.get_token()
                self.state = self.states.pop()
                return SequenceEndEvent(self.marks.pop(), token.end_mark)

        if self.check_token(KeyToken):
            token = self.peek_token()
            self.marks.append(token.start_mark)
            return self.parse_flow_sequence_entry_mapping_key()

        if self._flow_node_start():
            self.states.append(self.parse_flow_sequence_entry)
            return self.parse_flow_node()

        token = self.peek_token()
        raise ParserError("while parsing a flow sequence", self.marks[-1],
                          "expected the node content, but found %r" % token.id,
                          token.start_mark)

    def parse_flow_sequence_entry_mapping_key(self):
        token = self.get_token()
        self.state = self.parse_flow_sequence_entry_mapping_value
        return MappingStartEvent(None, None, True, token.start_mark,
                                 token.end_mark, flow_style=True)

    def parse_flow_sequence_entry_mapping_value(self):
        if self._flow_node_start():
            self.states.append(self.parse_flow_sequence_entry_mapping_end)
            return self.parse_flow_node()
        self.state = self.parse_flow_sequence_entry_mapping_end
        return self.process_empty_scalar(self.peek_token().start_mark)

    def parse_flow_sequence_entry_mapping_end(self):
        token = self.peek_token()
        self.state = self.parse_flow_sequence_entry
        return MappingEndEvent(self.marks.pop(), token.start_mark)

    def parse_flow_mapping_first_key(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        self.state = self.parse_flow_mapping_key
        return self.parse_flow_mapping_key(first=True)

    def parse_flow_mapping_key(self, first=False):
        if self.check_token(FlowMappingEndToken):
            token = self.get_token()
            self.state = self.states.pop()
            return MappingEndEvent(self.marks.pop(), token.end_mark)

        if not first:
            if self.check_token(FlowEntryToken):
                self.get_token()
            else:
                token = self.peek_token()
                raise ParserError(
                    "while parsing a flow mapping", self.marks[-1],
                    "expected ',' or '}', but got %r" % token.id,
                    token.start_mark)

            if self.check_token(FlowMappingEndToken):
                token = self.get_token()
                self.state = self.states.pop()
                return MappingEndEvent(self.marks.pop(), token.end_mark)

        if self.check_token(KeyToken):
            token = self.get_token()
            if self._flow_node_start():
                self.states.append(self.parse_flow_mapping_value)
                return self.parse_flow_node()
            self.state = self.parse_flow_mapping_value
            return self.process_empty_scalar(token.end_mark)

        if self._flow_node_start():
            self.states.append(self.parse_flow_mapping_value)
            return self.parse_flow_node()

        token = self.peek_token()
        raise ParserError("while parsing a flow mapping", self.marks[-1],
                          "expected the node content, but found %r" % token.id,
                          token.start_mark)

    def parse_flow_mapping_value(self):
        if self.check_token(ValueToken):
            token = self.get_token()
            if self._flow_node_start():
                self.states.append(self.parse_flow_mapping_key)
                return self.parse_flow_node()
            self.state = self.parse_flow_mapping_key
            return self.process_empty_scalar(token.end_mark)

        self.state = self.parse_flow_mapping_key
        return self.process_empty_scalar(self.peek_token().start_mark)


def tag_token_mark(tag, token, fallback):
    return fallback