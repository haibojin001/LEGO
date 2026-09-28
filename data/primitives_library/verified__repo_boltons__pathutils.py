import os
from os.path import expanduser, expandvars, join, normpath, split, splitext

__all__ = ['augpath', 'shrinkuser', 'expandpath']


def augpath(path, suffix='', prefix='', ext=None, base=None, dpath=None,
            multidot=False):
    directory, filename = split(path)

    if multidot:
        name_parts = filename.split('.', 1)
        original_base = name_parts[0]
        original_ext = '' if len(name_parts) == 1 else '.' + name_parts[1]
    else:
        original_base, original_ext = splitext(filename)

    if dpath is None:
        dpath = directory
    if base is None:
        base = original_base
    if ext is None:
        ext = original_ext

    return join(dpath, ''.join((prefix, base, suffix, ext)))


def shrinkuser(path, home='~'):
    normalized = normpath(path)
    user_home = expanduser('~')

    if normalized.startswith(user_home):
        if len(normalized) == len(user_home):
            normalized = home
        elif normalized[len(user_home)] == os.path.sep:
            normalized = home + normalized[len(user_home):]

    return normalized


def expandpath(path):
    return expandvars(expanduser(path))