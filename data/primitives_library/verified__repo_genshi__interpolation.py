from itertools import chain
import re
from tokenize import PseudoToken

from genshi.core import TEXT
from genshi.template.base import EXPR, TemplateSyntaxError
from genshi.template.eval import Expression

__all__ = ['interpolate']
__docformat__ = 'restructuredtext en'

NAMESTART = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
NAMECHARS = NAMESTART + '.0123456789'
PREFIX = '$'

_triple_quoted = r'[uU]?[rR]?("""|\'\'\')((?<!\\)\\\1|.)*?\1'
token_re = re.compile('(?s)(?:%s)|(?:%s)' % (_triple_quoted, PseudoToken))


def interpolate(text, filepath=None, lineno=-1, offset=0, lookup='strict'):
    """Yield text and expression events extracted from an interpolated string."""
    location = [filepath, lineno, offset]
    pending_text = []
    pending_position = None

    for expression, value in chain(lex(text, location, filepath), [(True, '')]):
        if expression:
            if pending_text:
                yield TEXT, ''.join(pending_text), pending_position
                pending_text[:] = []
                pending_position = None

            if value:
                try:
                    compiled = Expression(
                        value.strip(), location[0], location[1], lookup=lookup
                    )
                except SyntaxError as error:
                    raise TemplateSyntaxError(
                        error,
                        filepath,
                        location[1],
                        location[2] + (error.offset or 0),
                    )
                yield EXPR, compiled, tuple(location)
        else:
            pending_text.append(value)
            if pending_position is None:
                pending_position = tuple(location)

        if '\n' in value:
            pieces = value.splitlines()
            location[1] += len(pieces) - 1
            location[2] += len(pieces[-1])
        else:
            location[2] += len(value)


def lex(text, textpos, filepath):
    """Split a string into literal and interpolation-expression fragments."""
    cursor = 0
    marker = 0
    length = len(text)
    escaped = False

    while True:
        if escaped:
            marker = text.find(PREFIX, marker + 2)
            escaped = False
        else:
            marker = text.find(PREFIX, cursor)

        if marker < 0 or marker == length - 1:
            break

        following = text[marker + 1]

        if following == '{':
            if marker > cursor:
                yield False, text[cursor:marker]

            cursor = marker + 2
            nesting = 1

            while nesting:
                match = token_re.match(text, cursor)
                if match is None or not match.group():
                    raise TemplateSyntaxError(
                        'invalid syntax', filepath, *textpos[1:]
                    )

                cursor = match.end()
                start, end = match.regs[3]
                symbol = text[start:end]

                if symbol == '{':
                    nesting += 1
                elif symbol == '}':
                    nesting -= 1

            yield True, text[marker + 2:cursor - 1]

        elif following in NAMESTART:
            if marker > cursor:
                yield False, text[cursor:marker]
                cursor = marker

            cursor += 1
            while cursor < length and text[cursor] in NAMECHARS:
                cursor += 1

            yield True, text[marker + 1:cursor].strip()

        elif not escaped and following == PREFIX:
            if marker > cursor:
                yield False, text[cursor:marker]
            escaped = True
            cursor = marker + 1

        else:
            yield False, text[cursor:marker + 1]
            cursor = marker + 1

    if cursor < length:
        yield False, text[cursor:]