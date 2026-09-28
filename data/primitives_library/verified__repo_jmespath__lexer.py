import string
import warnings
from json import loads

from jmespath.exceptions import LexerError, EmptyExpressionError


class Lexer(object):
    START_IDENTIFIER = set(string.ascii_letters + '_')
    VALID_IDENTIFIER = set(string.ascii_letters + string.digits + '_')
    VALID_NUMBER = set(string.digits)
    WHITESPACE = set(' \t\n\r')
    SIMPLE_TOKENS = {
        '.': 'dot',
        '*': 'star',
        ']': 'rbracket',
        ',': 'comma',
        ':': 'colon',
        '@': 'current',
        '(': 'lparen',
        ')': 'rparen',
        '{': 'lbrace',
        '}': 'rbrace',
    }

    def tokenize(self, expression):
        self._initialize_for_expression(expression)
        while self._current is not None:
            char = self._current

            if char in self.SIMPLE_TOKENS:
                yield {
                    'type': self.SIMPLE_TOKENS[char],
                    'value': char,
                    'start': self._position,
                    'end': self._position + 1,
                }
                self._next()
            elif char in self.START_IDENTIFIER:
                start = self._position
                value = char
                while self._next() in self.VALID_IDENTIFIER:
                    value += self._current
                yield {
                    'type': 'unquoted_identifier',
                    'value': value,
                    'start': start,
                    'end': start + len(value),
                }
            elif char in self.WHITESPACE:
                self._next()
            elif char == '[':
                start = self._position
                following = self._next()
                if following == ']':
                    self._next()
                    yield {
                        'type': 'flatten',
                        'value': '[]',
                        'start': start,
                        'end': start + 2,
                    }
                elif following == '?':
                    self._next()
                    yield {
                        'type': 'filter',
                        'value': '[?',
                        'start': start,
                        'end': start + 2,
                    }
                else:
                    yield {
                        'type': 'lbracket',
                        'value': '[',
                        'start': start,
                        'end': start + 1,
                    }
            elif char == "'":
                yield self._consume_raw_string_literal()
            elif char == '|':
                yield self._match_or_else('|', 'or', 'pipe')
            elif char == '&':
                yield self._match_or_else('&', 'and', 'expref')
            elif char == '`':
                yield self._consume_literal()
            elif char in self.VALID_NUMBER:
                start = self._position
                text = self._consume_number()
                yield {
                    'type': 'number',
                    'value': int(text),
                    'start': start,
                    'end': start + len(text),
                }
            elif char == '-':
                start = self._position
                text = self._consume_number()
                if len(text) > 1:
                    yield {
                        'type': 'number',
                        'value': int(text),
                        'start': start,
                        'end': start + len(text),
                    }
                else:
                    raise LexerError(
                        lexer_position=start,
                        lexer_value=text,
                        message="Unknown token '%s'" % text,
                    )
            elif char == '"':
                yield self._consume_quoted_identifier()
            elif char == '<':
                yield self._match_or_else('=', 'lte', 'lt')
            elif char == '>':
                yield self._match_or_else('=', 'gte', 'gt')
            elif char == '!':
                yield self._match_or_else('=', 'ne', 'not')
            elif char == '=':
                if self._next() == '=':
                    yield {
                        'type': 'eq',
                        'value': '==',
                        'start': self._position - 1,
                        'end': self._position,
                    }
                    self._next()
                else:
                    if self._current is None:
                        position = self._position
                    else:
                        position = self._position - 1
                    raise LexerError(
                        lexer_position=position,
                        lexer_value='=',
                        message="Unknown token '='",
                    )
            else:
                raise LexerError(
                    lexer_position=self._position,
                    lexer_value=char,
                    message='Unknown token %s' % char,
                )

        yield {
            'type': 'eof',
            'value': '',
            'start': self._length,
            'end': self._length,
        }

    def _consume_number(self):
        value = self._current
        while self._next() in self.VALID_NUMBER:
            value += self._current
        return value

    def _initialize_for_expression(self, expression):
        if not expression:
            raise EmptyExpressionError()
        self._position = 0
        self._expression = expression
        self._chars = list(expression)
        self._length = len(expression)
        self._current = self._chars[0]

    def _next(self):
        if self._position == self._length - 1:
            self._current = None
        else:
            self._position += 1
            self._current = self._chars[self._position]
        return self._current

    def _consume_until(self, delimiter):
        beginning = self._position
        value = ''
        self._next()

        while self._current != delimiter:
            if self._current == '\\':
                value += '\\'
                self._next()
            if self._current is None:
                raise LexerError(
                    lexer_position=beginning,
                    lexer_value=self._expression[beginning:],
                    message='Unclosed %s delimiter' % delimiter,
                )
            value += self._current
            self._next()

        self._next()
        return value

    def _consume_literal(self):
        start = self._position
        text = self._consume_until('`').replace('\\`', '`')
        try:
            result = loads(text)
        except ValueError:
            try:
                result = loads('"%s"' % text.lstrip())
                warnings.warn(
                    'deprecated string literal syntax',
                    PendingDeprecationWarning,
                )
            except ValueError:
                raise LexerError(
                    lexer_position=start,
                    lexer_value=self._expression[start:],
                    message='Bad token %s' % text,
                )
        return {
            'type': 'literal',
            'value': result,
            'start': start,
            'end': self._position - start,
        }

    def _consume_quoted_identifier(self):
        start = self._position
        text = '"' + self._consume_until('"') + '"'
        try:
            return {
                'type': 'quoted_identifier',
                'value': loads(text),
                'start': start,
                'end': self._position - start,
            }
        except ValueError as error:
            raise LexerError(
                lexer_position=start,
                lexer_value=text,
                message=str(error).split(':')[0],
            )

    def _consume_raw_string_literal(self):
        start = self._position
        text = self._consume_until("'").replace("\\'", "'")
        return {
            'type': 'literal',
            'value': text,
            'start': start,
            'end': self._position - start,
        }

    def _match_or_else(self, expected, match_type, else_type):
        start = self._position
        char = self._current
        following = self._next()
        if following == expected:
            self._next()
            return {
                'type': match_type,
                'value': char + following,
                'start': start,
                'end': start + 1,
            }
        return {
            'type': else_type,
            'value': char,
            'start': start,
            'end': start,
        }