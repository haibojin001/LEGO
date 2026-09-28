from __future__ import annotations

from typing import NamedTuple

from tokenize_rt import ESCAPED_NL
from tokenize_rt import NON_CODING_TOKENS
from tokenize_rt import Offset
from tokenize_rt import Token
from tokenize_rt import UNIMPORTANT_WS


NEWLINES = frozenset({ESCAPED_NL, 'NEWLINE', 'NL'})
INDENT_TOKENS = frozenset({'INDENT', UNIMPORTANT_WS})
START_BRACES = frozenset({'(', '{', '['})
END_BRACES = frozenset({')', '}', ']'})


class Fix(NamedTuple):
    braces: tuple[int, int]
    multi_arg: bool
    remove_comma: bool
    initial_indent: int


def find_simple(first_brace: int, tokens: list[Token]) -> Fix | None:
    stack = [first_brace]
    has_multiple_arguments = False

    for position in range(first_brace + 1, len(tokens)):
        current = tokens[position]

        if current.name == 'OP' and current.src in START_BRACES:
            stack.append(position)
        elif current.name == 'OP' and current.src in END_BRACES:
            stack.pop()

        if len(stack) == 1 and current.src == ',':
            has_multiple_arguments = True

        if not stack:
            break
    else:
        raise AssertionError('Past end?')

    closing_brace = position
    same_line = tokens[first_brace].line == tokens[closing_brace].line

    if same_line and (
            tokens[closing_brace - 1].name == UNIMPORTANT_WS or
            tokens[closing_brace - 1].src == ','
    ):
        should_remove_comma = True
    elif same_line:
        return None
    else:
        should_remove_comma = False

    newline_position = first_brace
    while (
            newline_position >= 0 and
            tokens[newline_position].name not in NEWLINES
    ):
        newline_position -= 1

    if (
            newline_position >= 0 and
            tokens[newline_position + 1].name in INDENT_TOKENS
    ):
        indent = len(tokens[newline_position + 1].src)
    else:
        indent = 0

    return Fix(
        braces=(first_brace, closing_brace),
        multi_arg=has_multiple_arguments,
        remove_comma=should_remove_comma,
        initial_indent=indent,
    )


def find_call(
        arg_offsets: set[Offset],
        i: int,
        tokens: list[Token],
) -> Fix | None:
    opening_parens: list[int] = []
    call_brace = None

    for position in range(i, len(tokens)):
        current = tokens[position]

        if current.name == 'OP' and current.src == '(':
            opening_parens.append(position)
        elif (
                current.name == 'OP' and
                current.src == ')' and
                opening_parens
        ):
            opening_parens.pop()

        if (current.line, current.utf8_byte_offset) in arg_offsets:
            call_brace = opening_parens[0]
            break
    else:
        raise AssertionError('Past end?')

    return find_simple(call_brace, tokens)


def fix_brace(
        tokens: list[Token],
        fix_data: Fix | None,
        add_comma: bool,
        remove_comma: bool,
) -> None:
    if fix_data is None:
        return

    opening, closing = fix_data.braces
    opening_hugs = tokens[opening + 1].name not in NON_CODING_TOKENS
    closing_hugs = tokens[closing - 1].name not in NON_CODING_TOKENS

    preserve_hugging = (
        (
            not fix_data.multi_arg and
            tokens[opening + 1].src in START_BRACES and
            tokens[closing - 1].src in END_BRACES
        ) or
        opening + 2 == closing or
        (
            tokens[opening + 1].name == 'FSTRING_START' and
            tokens[closing - 1].name == 'FSTRING_END'
        ) or
        (
            tokens[opening + 1].name == 'TSTRING_START' and
            tokens[closing - 1].name == 'TSTRING_END'
        ) or
        fix_data.remove_comma
    )
    if preserve_hugging:
        opening_hugs = False
        closing_hugs = False

    if opening_hugs:
        new_indent = fix_data.initial_indent + 4
        tokens[opening + 1:opening + 1] = [
            Token('NL', '\n'),
            Token(UNIMPORTANT_WS, ' ' * new_indent),
        ]
        closing += 2

        smallest_indent = None
        indentation_tokens: list[int] = []
        missing_indentation: list[int] = []

        for position in range(opening + 3, closing):
            if (
                    tokens[position - 1].name == 'NL' and
                    tokens[position].name != 'NL'
            ):
                if tokens[position].name == UNIMPORTANT_WS:
                    width = len(tokens[position].src)
                    if smallest_indent is None:
                        smallest_indent = width
                    elif width < smallest_indent:
                        smallest_indent = width
                    indentation_tokens.append(position)
                else:
                    smallest_indent = 0
                    missing_indentation.append(position)

        if indentation_tokens:
            assert smallest_indent is not None
            for position in indentation_tokens:
                old_width = len(tokens[position].src)
                adjusted_width = old_width - smallest_indent + new_indent
                tokens[position] = tokens[position]._replace(
                    src=' ' * adjusted_width,
                )

        for position in reversed(missing_indentation):
            tokens.insert(position, Token(UNIMPORTANT_WS, ' ' * new_indent))
            closing += 1

    if closing_hugs:
        tokens[closing:closing] = [
            Token('NL', '\n'),
            Token(UNIMPORTANT_WS, ' ' * fix_data.initial_indent),
        ]
        closing += 2

    previous = closing - 1
    while tokens[previous].name in NON_CODING_TOKENS:
        previous -= 1

    if (
            add_comma and
            tokens[previous].src != ',' and
            previous + 1 != closing
    ):
        tokens.insert(previous + 1, Token('OP', ','))

    before_closing = tokens[closing - 1]
    before_before_closing = tokens[closing - 2]
    if (
            before_closing.name == UNIMPORTANT_WS and
            before_before_closing.name == 'NL' and
            len(before_closing.src) != fix_data.initial_indent
    ):
        tokens[closing - 1] = before_closing._replace(
            src=' ' * fix_data.initial_indent,
        )

    if fix_data.remove_comma:
        deletion_start = closing
        if tokens[deletion_start - 1].name == UNIMPORTANT_WS:
            deletion_start -= 1
        if remove_comma and tokens[deletion_start - 1].src == ',':
            deletion_start -= 1
        del tokens[deletion_start:closing]