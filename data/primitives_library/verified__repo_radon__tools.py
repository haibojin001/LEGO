import fnmatch
import hashlib
import json
import locale
import os
import platform
import re
import sys
import xml.etree.cElementTree as et
from contextlib import contextmanager

from radon.cli.colors import BRIGHT, LETTERS_COLORS, RANKS_COLORS, RESET, TEMPLATE
from radon.complexity import cc_rank
from radon.visitors import Function

try:
    import nbformat

    SUPPORTS_IPYNB = True
except ImportError:
    SUPPORTS_IPYNB = False


if platform.python_implementation() == 'PyPy':

    @contextmanager
    def _open(path):
        if path == '-':
            yield sys.stdin
        else:
            with open(path) as stream:
                yield stream

else:
    default_encoding = (
        'utf-8'
        if sys.version_info[:2] >= (3, 0)
        else locale.getpreferredencoding(False)
    )
    _encoding = os.getenv('RADONFILESENCODING', default_encoding)
    _open_function = open

    @contextmanager
    def _open(path):
        if path == '-':
            yield sys.stdin
        else:
            with _open_function(path, encoding=_encoding) as stream:
                yield stream


def _is_python_file(filename):
    if (
        filename == '-'
        or filename.endswith('.py')
        or (SUPPORTS_IPYNB and filename.endswith('.ipynb'))
    ):
        return True

    try:
        with open(filename) as stream:
            first_line = stream.readline()
    except Exception:
        return False

    return first_line.startswith('#!') and 'python' in first_line


def filter_out(strings, patterns):
    patterns = patterns or ()
    for string in strings:
        if not any(fnmatch.fnmatch(string, pattern) for pattern in patterns):
            yield string


def explore_directories(start, exclude, ignore):
    exclude = exclude or ()
    ignore = ignore or ()

    for root, dirs, files in os.walk(start):
        dirs[:] = list(filter_out(dirs, ignore))

        if any(re.match(pattern, root) for pattern in exclude):
            dirs[:] = []
            continue

        for filename in files:
            fullpath = os.path.join(root, filename)
            if any(re.match(pattern, fullpath) for pattern in exclude):
                continue
            if any(fnmatch.fnmatch(filename, pattern) for pattern in ignore):
                continue
            if _is_python_file(fullpath):
                yield fullpath


def iter_filenames(paths, exclude=None, ignore=None):
    if exclude is None:
        exclude_patterns = ()
    elif isinstance(exclude, str):
        exclude_patterns = tuple(
            pattern for pattern in exclude.split(',') if pattern
        )
    else:
        exclude_patterns = tuple(exclude)

    if ignore is None:
        ignore_patterns = ()
    elif isinstance(ignore, str):
        ignore_patterns = tuple(pattern for pattern in ignore.split(',') if pattern)
    else:
        ignore_patterns = tuple(ignore)

    for path in paths:
        if os.path.isfile(path):
            if (
                _is_python_file(path)
                and not any(re.match(pattern, path) for pattern in exclude_patterns)
                and not any(
                    fnmatch.fnmatch(os.path.basename(path), pattern)
                    for pattern in ignore_patterns
                )
            ):
                yield path
        elif os.path.isdir(path):
            yield from explore_directories(path, exclude_patterns, ignore_patterns)


def read_file(path):
    with _open(path) as stream:
        return stream.read()


def get_content(path):
    content = read_file(path)

    if not (SUPPORTS_IPYNB and path.endswith('.ipynb')):
        return content

    notebook = nbformat.reads(content, as_version=4)
    return '\n'.join(
        cell.source
        for cell in notebook.cells
        if cell.get('cell_type') == 'code'
    )


def get_module_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def cc_to_dict(obj):
    result = {
        'type': obj.letter,
        'rank': cc_rank(obj.complexity),
        'complexity': obj.complexity,
        'lineno': obj.lineno,
        'col_offset': obj.col_offset,
        'endline': obj.endline,
        'name': obj.fullname,
        'closures': [cc_to_dict(closure) for closure in obj.closures],
    }

    if isinstance(obj, Function) and getattr(obj, 'classname', None):
        result['classname'] = obj.classname

    return result


def raw_to_dict(raw):
    if hasattr(raw, '_asdict'):
        return raw._asdict()

    names = (
        'loc',
        'lloc',
        'sloc',
        'comments',
        'multi',
        'blank',
        'single_comments',
    )
    return {
        name: getattr(raw, name)
        for name in names
        if hasattr(raw, name)
    }


def mi_to_dict(mi):
    rank = 'A' if mi >= 20 else 'B' if mi >= 10 else 'C'
    return {'mi': mi, 'rank': rank}


def _xml_measure(parent, filename, block):
    measure = et.SubElement(parent, 'measure')
    values = (
        ('complexity', block.get('complexity')),
        ('unit', block.get('name')),
        ('classification', block.get('rank')),
        ('file', filename),
        ('startLineNumber', block.get('lineno')),
        ('endLineNumber', block.get('endline')),
    )

    for tag, value in values:
        child = et.SubElement(measure, tag)
        child.text = '' if value is None else str(value)

    for closure in block.get('closures', ()):
        _xml_measure(parent, filename, closure)


def dict_to_xml(data):
    root = et.Element('ccm')

    for filename, blocks in data.items():
        if isinstance(blocks, dict):
            blocks = (blocks,)
        for block in blocks:
            _xml_measure(root, filename, block)

    return root


def get_fingerprint(*args):
    payload = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def log(msg, *args, **kwargs):
    stream = kwargs.pop('stream', sys.stderr)
    if kwargs:
        msg = msg.format(*args, **kwargs)
    else:
        msg = msg.format(*args)
    stream.write(msg + '\n')


def format_cc(issue, show_complexity=False):
    if isinstance(issue, dict):
        letter = issue.get('type', '')
        lineno = issue.get('lineno', 0)
        col_offset = issue.get('col_offset', 0)
        name = issue.get('name', '')
        rank = issue.get('rank', '')
        complexity = issue.get('complexity', '')
    else:
        letter = issue.letter
        lineno = issue.lineno
        col_offset = issue.col_offset
        name = issue.fullname
        complexity = issue.complexity
        rank = cc_rank(complexity)

    colored_letter = '{}{}{}'.format(LETTERS_COLORS.get(letter, ''), letter, RESET)
    colored_rank = '{}{}{}'.format(RANKS_COLORS.get(rank, ''), rank, RESET)
    complexity_text = complexity if show_complexity else rank

    try:
        return TEMPLATE.format(
            letter=colored_letter,
            lineno=lineno,
            col_offset=col_offset,
            name=name,
            rank=colored_rank,
            complexity=complexity_text,
            bright=BRIGHT,
            reset=RESET,
        )
    except (IndexError, KeyError):
        return '{} {}:{} {} - {} ({})'.format(
            colored_letter,
            lineno,
            col_offset,
            name,
            colored_rank,
            complexity,
        )