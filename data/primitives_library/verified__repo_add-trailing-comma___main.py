from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from collections.abc import Sequence

from tokenize_rt import Token
from tokenize_rt import src_to_tokens
from tokenize_rt import tokens_to_src

from add_trailing_comma._ast_helpers import ast_parse
from add_trailing_comma._data import FUNCS
from add_trailing_comma._data import visit
from add_trailing_comma._token_helpers import START_BRACES
from add_trailing_comma._token_helpers import find_simple
from add_trailing_comma._token_helpers import fix_brace


def _changing_list(lst: list[Token]) -> Iterable[tuple[int, Token]]:
    index = 0
    while index < len(lst):
        item = lst[index]
        yield index, item
        index += 1


def _fix_src(contents_text: str) -> str:
    try:
        tree = ast_parse(contents_text)
    except SyntaxError:
        return contents_text

    callbacks_by_offset = visit(FUNCS, tree)
    tokens = src_to_tokens(contents_text)

    for index, token in _changing_list(tokens):
        if token.src == '':
            continue

        for callback in callbacks_by_offset.get(token.offset, ()):
            callback(index, tokens)

        if token.name == 'OP' and token.src in START_BRACES:
            simple = find_simple(index, tokens)
            fix_brace(
                tokens,
                simple,
                add_comma=False,
                remove_comma=False,
            )

    return tokens_to_src(tokens)


def fix_file(filename: str, args: argparse.Namespace) -> int:
    if filename == '-':
        data = sys.stdin.buffer.read()
    else:
        with open(filename, 'rb') as f:
            data = f.read()

    try:
        original = data.decode()
    except UnicodeDecodeError:
        print(
            f'{filename} is non-utf-8 (not supported)',
            file=sys.stderr,
        )
        return 1

    fixed = _fix_src(original)

    if filename == '-':
        print(fixed, end='')
    elif fixed != original:
        print(f'Rewriting {filename}', file=sys.stderr)
        with open(filename, 'wb') as f:
            f.write(fixed.encode())

    if args.exit_zero_even_if_changed:
        return 0
    return fixed != original


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument(
        '--exit-zero-even-if-changed',
        action='store_true',
    )
    args = parser.parse_args(argv)

    status = 0
    for filename in args.filenames:
        status |= fix_file(filename, args)
    return status


if __name__ == '__main__':
    raise SystemExit(main())