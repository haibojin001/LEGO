import collections
import operator
import tokenize

try:
    import StringIO as io
except ImportError:
    import io


__all__ = [
    'OP',
    'COMMENT',
    'TOKEN_NUMBER',
    'NL',
    'NEWLINE',
    'EM',
    'Module',
    '_generate',
    '_fewer_tokens',
    '_find',
    '_logical',
    'analyze',
]


COMMENT = tokenize.COMMENT
OP = tokenize.OP
NL = tokenize.NL
NEWLINE = tokenize.NEWLINE
EM = tokenize.ENDMARKER

TOKEN_NUMBER = operator.itemgetter(0)

Module = collections.namedtuple(
    'Module',
    ('loc', 'lloc', 'sloc', 'comments', 'multi', 'blank', 'single_comments'),
)


def _generate(code):
    return list(tokenize.generate_tokens(io.StringIO(code).readline))


def _fewer_tokens(tokens, remove):
    for token in tokens:
        if token[0] not in remove:
            yield token


def _find(tokens, token, value):
    for reverse_index, candidate in enumerate(reversed(tokens)):
        if candidate[:2] == (token, value):
            return len(tokens) - reverse_index - 1
    raise ValueError('(token, value) pair not found')


def _split_tokens(tokens, token, value):
    groups = [[]]
    separator = (token, value)
    for candidate in tokens:
        if candidate[:2] == separator:
            groups.append([])
        else:
            groups[-1].append(candidate)
    return groups


def _get_all_tokens(line, lines):
    text = line
    collected = [line]

    while True:
        try:
            result = _generate(text)
        except tokenize.TokenError:
            result = None

        if result is not None and not any(
            token[0] == tokenize.ERRORTOKEN for token in result
        ):
            return result, collected

        following = next(lines)
        text += '\n' + following
        collected.append(following)


def _logical(tokens):
    def count_group(group):
        meaningful = list(_fewer_tokens(group, (COMMENT, NL, NEWLINE)))

        try:
            colon_position = _find(meaningful, OP, ':')
        except ValueError:
            remaining = list(_fewer_tokens(meaningful, (NL, NEWLINE, EM)))
            return 1 if remaining else 0

        return 1 if colon_position == len(meaningful) - 2 else 2

    return sum(count_group(group) for group in _split_tokens(tokens, OP, ';'))


def is_single_token(token_number, tokens):
    return (
        TOKEN_NUMBER(tokens[0]) == token_number
        and all(TOKEN_NUMBER(token) in (EM, NL, NEWLINE) for token in tokens[1:])
    )


def analyze(source):
    lloc = 0
    comments = 0
    single_comments = 0
    multi = 0
    blank = 0
    sloc = 0

    lines = (line.strip() for line in source.splitlines())
    lineno = 1

    for line in lines:
        try:
            tokens, parsed_lines = _get_all_tokens(line, lines)
        except StopIteration:
            raise SyntaxError('SyntaxError at line: {0}'.format(lineno))

        lineno += len(parsed_lines)

        comments += sum(
            1 for token in tokens if TOKEN_NUMBER(token) == tokenize.COMMENT
        )

        if is_single_token(tokenize.COMMENT, tokens):
            single_comments += 1
        elif is_single_token(tokenize.STRING, tokens):
            _, _, start, end, _ = tokens[0]
            if start[0] == end[0]:
                single_comments += 1
            else:
                multi += sum(1 for item in parsed_lines if item)
                blank += sum(1 for item in parsed_lines if not item)
        else:
            for parsed_line in parsed_lines:
                if parsed_line:
                    sloc += 1
                else:
                    blank += 1

        lloc += _logical(tokens)

    loc = sloc + blank + multi + single_comments
    return Module(loc, lloc, sloc, comments, multi, blank, single_comments)