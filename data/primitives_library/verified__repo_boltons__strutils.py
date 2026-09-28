import builtins
import collections
import itertools
import re
import shlex
import string
import subprocess
import sys
import typing
import unicodedata
import uuid
import zlib
from collections.abc import Mapping
from gzip import GzipFile
from html import entities as htmlentitydefs
from html.parser import HTMLParser
from io import BytesIO as StringIO

__all__ = [
    'camel2under', 'under2camel', 'slugify', 'split_punct_ws',
    'unit_len', 'ordinalize', 'cardinalize', 'pluralize', 'singularize',
    'asciify', 'is_ascii', 'is_uuid', 'html2text', 'strip_ansi',
    'bytes2human', 'find_hashtags', 'a10n', 'gzip_bytes', 'gunzip_bytes',
    'iter_splitlines', 'indent', 'escape_shell_args',
    'args2cmd', 'args2sh', 'parse_int_list', 'format_int_list',
    'complement_int_list', 'int_ranges_from_int_list', 'MultiReplace',
    'multi_replace', 'unwrap_text', 'removeprefix',
    'human_readable_list', 'ellipsize'
]

_punct_ws_str = string.punctuation + string.whitespace
_punct_re = re.compile('[' + re.escape(_punct_ws_str) + ']+')
_camel2under_re = re.compile(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
_ansi_re = re.compile(
    r'\x1B(?:'
    r'[\x30-\x3F]*[\x20-\x2F]*[\x40-\x7E]'
    r'|\][^\x1B\x07]*(?:\x07|\x1B\\)'
    r')'
)
_hashtag_re = re.compile(r'(?<!\w)#(\w*[^\W_]\w*)', re.UNICODE)

_IRR_S2P = {
    'addendum': 'addenda',
    'analysis': 'analyses',
    'antenna': 'antennae',
    'appendix': 'appendices',
    'axis': 'axes',
    'bacterium': 'bacteria',
    'basis': 'bases',
    'child': 'children',
    'chassis': 'chassis',
    'codex': 'codices',
    'corpus': 'corpora',
    'criterion': 'criteria',
    'curriculum': 'curricula',
    'datum': 'data',
    'deer': 'deer',
    'diagnosis': 'diagnoses',
    'ellipsis': 'ellipses',
    'fish': 'fish',
    'focus': 'foci',
    'foot': 'feet',
    'formula': 'formulae',
    'fungus': 'fungi',
    'genus': 'genera',
    'goose': 'geese',
    'half': 'halves',
    'hypothesis': 'hypotheses',
    'index': 'indices',
    'knife': 'knives',
    'leaf': 'leaves',
    'life': 'lives',
    'loaf': 'loaves',
    'man': 'men',
    'matrix': 'matrices',
    'means': 'means',
    'medium': 'media',
    'memorandum': 'memoranda',
    'moose': 'moose',
    'mouse': 'mice',
    'nucleus': 'nuclei',
    'oasis': 'oases',
    'ovum': 'ova',
    'ox': 'oxen',
    'parenthesis': 'parentheses',
    'person': 'people',
    'phenomenon': 'phenomena',
    'phylum': 'phyla',
    'potato': 'potatoes',
    'radius': 'radii',
    'self': 'selves',
    'series': 'series',
    'sheep': 'sheep',
    'species': 'species',
    'stimulus': 'stimuli',
    'syllabus': 'syllabi',
    'synopsis': 'synopses',
    'thesis': 'theses',
    'tooth': 'teeth',
    'vertex': 'vertices',
    'vita': 'vitae',
    'wife': 'wives',
    'wolf': 'wolves',
}
_IRR_P2S = {plural: singular for singular, plural in _IRR_S2P.items()}

_ORDINAL_MAP = {'1': 'st', '2': 'nd', '3': 'rd'}

_ASCII_REPLACEMENTS = {
    '\u00a0': ' ',
    '\u00a1': '!',
    '\u00a2': 'cents',
    '\u00a3': 'GBP',
    '\u00a4': 'currency',
    '\u00a5': 'JPY',
    '\u00a6': '|',
    '\u00a7': 'SS',
    '\u00a8': '"',
    '\u00a9': '(c)',
    '\u00aa': 'a',
    '\u00ab': '"',
    '\u00ac': 'not',
    '\u00ad': '-',
    '\u00ae': '(R)',
    '\u00af': '-',
    '\u00b0': ' degrees ',
    '\u00b1': '+/-',
    '\u00b2': '^2',
    '\u00b3': '^3',
    '\u00b4': "'",
    '\u00b5': 'u',
    '\u00b6': 'P',
    '\u00b7': '*',
    '\u00b8': ',',
    '\u00b9': '^1',
    '\u00ba': 'o',
    '\u00bb': '"',
    '\u00bc': '1/4',
    '\u00bd': '1/2',
    '\u00be': '3/4',
    '\u00bf': '?',
    '\u00c6': 'AE',
    '\u00d0': 'D',
    '\u00d7': 'x',
    '\u00d8': 'O',
    '\u00de': 'Th',
    '\u00df': 'ss',
    '\u00e6': 'ae',
    '\u00f0': 'd',
    '\u00f7': '/',
    '\u00f8': 'o',
    '\u00fe': 'th',
    '\u0110': 'D',
    '\u0111': 'd',
    '\u0126': 'H',
    '\u0127': 'h',
    '\u0131': 'i',
    '\u0132': 'IJ',
    '\u0133': 'ij',
    '\u0138': 'k',
    '\u013f': 'L',
    '\u0140': 'l',
    '\u0141': 'L',
    '\u0142': 'l',
    '\u014a': 'N',
    '\u014b': 'n',
    '\u0152': 'OE',
    '\u0153': 'oe',
    '\u0166': 'T',
    '\u0167': 't',
    '\u017f': 's',
    '\u0192': 'f',
    '\u02c6': '^',
    '\u02dc': '~',
    '\u2013': '-',
    '\u2014': '--',
    '\u2018': "'",
    '\u2019': "'",
    '\u201a': ',',
    '\u201b': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u201e': '"',
    '\u2020': '+',
    '\u2021': '++',
    '\u2022': '*',
    '\u2026': '...',
    '\u2030': ' per mille ',
    '\u2039': '<',
    '\u203a': '>',
    '\u20ac': 'EUR',
    '\u2122': '(TM)',
    '\u2190': '<-',
    '\u2191': '^',
    '\u2192': '->',
    '\u2193': 'v',
    '\u2212': '-',
    '\u221e': 'infinity',
    '\u2260': '!=',
    '\u2264': '<=',
    '\u2265': '>=',
    '\u00c4': 'Ae',
    '\u00d6': 'Oe',
    '\u00dc': 'Ue',
    '\u00e4': 'ae',
    '\u00f6': 'oe',
    '\u00fc': 'ue',
}


def camel2under(camel_string):
    return _camel2under_re.sub(r'_\1', camel_string).lower()


def under2camel(under_string):
    return ''.join(part.capitalize() or '_' for part in under_string.split('_'))


def split_punct_ws(text):
    return [part for part in _punct_re.split(text) if part]


def slugify(text, delim='_', lower=True, ascii=False):
    ret = delim.join(split_punct_ws(text)) or delim if text else ''
    if ascii:
        ret = asciify(ret)
    if lower:
        ret = ret.lower()
    return ret


def unit_len(sized_iterable, unit_noun='item'):
    count = len(sized_iterable)
    units = cardinalize(unit_noun, count)
    if count:
        return '{} {}'.format(count, units)
    return 'No {}'.format(units)


def ordinalize(number, ext_only=False):
    numstr = str(number)
    ext = ''
    if numstr and numstr[-1] in string.digits:
        if len(numstr) > 1 and numstr[-2] == '1':
            ext = 'th'
        else:
            ext = _ORDINAL_MAP.get(numstr[-1], 'th')
    return ext if ext_only else numstr + ext


def _match_case(master, disciple):
    if not master.strip():
        return disciple
    if master.lower() == master:
        return disciple.lower()
    if master.upper() == master:
        return disciple.upper()
    if master.title() == master:
        return disciple.title()
    return disciple


def singularize(word):
    orig_word = word
    word = word.strip().lower()
    if not word or word in _IRR_S2P:
        return orig_word

    irregular = _IRR_P2S.get(word)
    if irregular:
        singular = irregular
    elif not word.endswith('s'):
        return orig_word
    elif len(word) == 2:
        singular = word[:-1]
    elif word.endswith('ies') and word[-4:-3] not in 'aeiou':
        singular = word[:-3] + 'y'
    elif word.endswith('es') and word[-3] == 's':
        singular = word[:-2]
    elif word.endswith('ss'):
        return orig_word
    else:
        singular = word[:-1]
    return _match_case(orig_word, singular)


def pluralize(word):
    orig_word = word
    word = word.strip().lower()
    if not word or word in _IRR_P2S:
        return orig_word

    irregular = _IRR_S2P.get(word)
    if irregular:
        plural = irregular
    elif word.endswith('y') and word[-2:-1] not in 'aeiou':
        plural = word[:-1] + 'ies'
    elif word[-1] in 'sx' or word.endswith('ch') or word.endswith('sh'):
        plural = word if word.endswith('es') else word + 'es'
    else:
        plural = word + 's'
    return _match_case(orig_word, plural)


def cardinalize(unit_noun, count):
    if count == 1:
        return unit_noun
    return pluralize(unit_noun)


def asciify(text, ignore=False):
    if isinstance(text, bytes):
        text.decode('ascii')
        return text
    try:
        return text.encode('ascii')
    except UnicodeEncodeError:
        pass
    text = ''.join(_ASCII_REPLACEMENTS.get(char, char) for char in text)
    text = unicodedata.normalize('NFKD', text)
    errors = 'ignore' if ignore else 'replace'
    return text.encode('ascii', errors)


def is_ascii(text):
    try:
        if isinstance(text, bytes):
            text.decode('ascii')
        else:
            text.encode('ascii')
    except (UnicodeError, AttributeError):
        return False
    return True


def is_uuid(obj, version=4):
    try:
        return uuid.UUID(str(obj)).version == version
    except (ValueError, AttributeError, TypeError):
        return False


class _HTMLTextParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def html2text(html):
    parser = _HTMLTextParser()
    parser.feed(html)
    parser.close()
    return ''.join(parser.parts)


def strip_ansi(text):
    return _ansi_re.sub('', text)


def bytes2human(nbytes, ndigits=0):
    suffixes = ('B', 'K', 'M', 'G', 'T', 'P', 'E')
    nbytes = float(nbytes)
    index = 0
    while abs(nbytes) >= 1024 and index < len(suffixes) - 1:
        nbytes /= 1024.0
        index += 1
    return ('{0:.%df}{1}' % ndigits).format(nbytes, suffixes[index])


def find_hashtags(string):
    return _hashtag_re.findall(string)


def a10n(string):
    if len(string) < 3:
        return string
    return '{}{}{}'.format(string[0], len(string) - 2, string[-1])


def gzip_bytes(bytestring, compresslevel=9):
    buffer = StringIO()
    with GzipFile(fileobj=buffer, mode='wb', compresslevel=compresslevel) as gzip_file:
        gzip_file.write(bytestring)
    return buffer.getvalue()


def gunzip_bytes(bytestring):
    buffer = StringIO(bytestring)
    with GzipFile(fileobj=buffer, mode='rb') as gzip_file:
        return gzip_file.read()


def iter_splitlines(text):
    for line in text.splitlines():
        yield line


def indent(text, margin, newline='\n', key=bool):
    return newline.join(margin + line if key(line) else line
                        for line in text.split(newline))


def args2cmd(args, sep=' '):
    return sep.join(subprocess.list2cmdline([arg]) for arg in args)


def args2sh(args, sep=' '):
    return sep.join(shlex.quote(arg) for arg in args)


def escape_shell_args(args, sep=' ', style=None):
    if style is None:
        style = 'cmd' if sys.platform == 'win32' else 'sh'
    if style == 'cmd':
        return args2cmd(args, sep)
    if style == 'sh':
        return args2sh(args, sep)
    raise ValueError('unrecognized shell style: {!r}'.format(style))


_INT_RANGE_RE = re.compile(r'^\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*$')


def parse_int_list(range_string, sep=',', range_sep='-'):
    if not range_string:
        return []
    ret = []
    for item in range_string.split(sep):
        item = item.strip()
        if not item:
            continue
        match = _INT_RANGE_RE.match(item) if range_sep == '-' else None
        if match:
            start, end = map(int, match.groups())
            step = 1 if start <= end else -1
            ret.extend(range(start, end + step, step))
        elif range_sep in item:
            start, end = (int(v.strip()) for v in item.split(range_sep, 1))
            step = 1 if start <= end else -1
            ret.extend(range(start, end + step, step))
        else:
            ret.append(int(item))
    return ret


def int_ranges_from_int_list(int_list):
    ints = sorted(set(int_list))
    ranges = []
    for _, group in itertools.groupby(enumerate(ints), lambda pair: pair[1] - pair[0]):
        group = list(group)
        ranges.append((group[0][1], group[-1][1]))
    return ranges


def format_int_list(int_list, delim=', ', range_delim='-'):
    chunks = []
    for start, end in int_ranges_from_int_list(int_list):
        if start == end:
            chunks.append(str(start))
        else:
            chunks.append('{}{}{}'.format(start, range_delim, end))
    return delim.join(chunks)


def complement_int_list(range_string, range_start=0, range_end=None,
                        sep=',', range_sep='-'):
    ints = parse_int_list(range_string, sep=sep, range_sep=range_sep)
    if range_end is None:
        range_end = max(ints)
    excluded = set(ints)
    complement = [value for value in range(range_start, range_end + 1)
                  if value not in excluded]
    return format_int_list(complement, delim=sep, range_delim=range_sep)


class MultiReplace:
    def __init__(self, sub_map, **kwargs):
        self.sub_map = sub_map
        keys = sorted(sub_map, key=len, reverse=True)
        self.combined_pattern = re.compile(
            '|'.join(re.escape(key) for key in keys), **kwargs)

    def _get_replacement(self, match):
        return self.sub_map[match.group(0)]

    def sub(self, text):
        return self.combined_pattern.sub(self._get_replacement, text)


def multi_replace(text, sub_map, **kwargs):
    return MultiReplace(sub_map, **kwargs).sub(text)


def unwrap_text(text, ending='\n'):
    paragraphs = []
    current = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(' '.join(current))
            current = []
    if current:
        paragraphs.append(' '.join(current))
    return (ending * 2).join(paragraphs)


def removeprefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def human_readable_list(iterable, conjunction='and', oxford=True):
    items = [str(item) for item in iterable]
    length = len(items)
    if not length:
        return ''
    if length == 1:
        return items[0]
    if length == 2:
        return '{} {} {}'.format(items[0], conjunction, items[1])
    final_sep = ', {} '.format(conjunction) if oxford else ' {} '.format(conjunction)
    return ', '.join(items[:-1]) + final_sep + items[-1]


def ellipsize(text, limit, ellipsis='...', pos=0.5):
    if len(text) <= limit:
        return text
    if limit <= len(ellipsis):
        return ellipsis[:limit]
    if not 0 <= pos <= 1:
        raise ValueError('pos must be between 0 and 1')
    available = limit - len(ellipsis)
    left = int(available * pos)
    right = available - left
    if right:
        return text[:left] + ellipsis + text[-right:]
    return text[:left] + ellipsis