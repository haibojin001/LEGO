import re
from operator import methodcaller

from .primitives import EMPTY


__all__ = [
    're_iter', 're_all', 're_find', 're_finder', 're_test', 're_tester',
    'str_join',
    'cut_prefix', 'cut_suffix',
]


_re_type = type(re.compile(''))


def _make_getter(pattern):
    group_count = pattern.groups
    named_groups = pattern.groupindex

    if group_count == 0:
        return methodcaller('group')
    if group_count == 1 and not named_groups:
        return methodcaller('group', 1)
    if not named_groups:
        return methodcaller('groups')
    if group_count == len(named_groups):
        return methodcaller('groupdict')
    return lambda match: match


def _prepare(regex, flags):
    if not isinstance(regex, _re_type):
        regex = re.compile(regex, flags)
    return regex, _make_getter(regex)


def re_iter(regex, s, flags=0):
    """Iterate over regex matches in their most convenient representation."""
    pattern, extract = _prepare(regex, flags)
    return map(extract, pattern.finditer(s))


def re_all(regex, s, flags=0):
    """Return all regex matches in their most convenient representation."""
    return list(re_iter(regex, s, flags))


def re_find(regex, s, flags=0):
    """Find the first regex match in a string, or return None."""
    return re_finder(regex, flags)(s)


def re_finder(regex, flags=0):
    """Build a function that finds the first match of a regex."""
    pattern, extract = _prepare(regex, flags)

    def finder(s):
        match = pattern.search(s)
        return None if match is None else extract(match)

    return finder


def re_test(regex, s, flags=0):
    """Test whether a regex occurs in a string."""
    return re_tester(regex, flags)(s)


def re_tester(regex, flags=0):
    """Build a predicate that tests whether a regex occurs in a string."""
    if not isinstance(regex, _re_type):
        regex = re.compile(regex, flags)

    def tester(s):
        return bool(regex.search(s))

    return tester


def str_join(sep, seq=EMPTY):
    """Join values after converting each value to the separator's string type."""
    if seq is EMPTY:
        return str_join('', sep)
    return sep.join(map(sep.__class__, seq))


def cut_prefix(s, prefix):
    """Remove a leading prefix when it is present."""
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def cut_suffix(s, suffix):
    """Remove a trailing suffix when it is present."""
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s