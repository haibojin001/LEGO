import os
import re
import sys
import stat
import errno
import fnmatch
from shutil import copy2, copystat, Error


__all__ = ['mkdir_p', 'atomic_save', 'AtomicSaver', 'FilePerms',
           'iter_find_files', 'copytree']


FULL_PERMS = 0o777
RW_PERMS = 0o666
_SINGLE_FULL_PERM = 0o7


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return
        raise
    return


class FilePerms:
    class _FilePermProperty:
        _perm_chars = 'rwx'
        _perm_set = frozenset(_perm_chars)
        _perm_val = {'r': 4, 'w': 2, 'x': 1}

        def __init__(self, attribute, offset):
            self.attribute = attribute
            self.offset = offset

        def __get__(self, obj, owner=None):
            if obj is None:
                return self
            return getattr(obj, self.attribute)

        def __set__(self, obj, value):
            old_value = getattr(obj, self.attribute)
            if old_value == value:
                return

            try:
                bad_chars = set(str(value)) - self._perm_set
            except TypeError:
                raise TypeError('expected string, not %r' % value)

            if bad_chars:
                raise ValueError(
                    'got invalid chars %r in permission specification %r, '
                    'expected empty string or one or more of %r'
                    % (bad_chars, value, self._perm_chars))

            normalized = ''.join(sorted(
                set(value),
                key=lambda char: self._perm_val[char],
                reverse=True))
            setattr(obj, self.attribute, normalized)

            bits = 0
            bit_names = 'xwr'
            for char in normalized:
                bits |= 2 ** bit_names.index(char)
            obj._integer |= bits << (self.offset * 3)

    def __init__(self, user='', group='', other=''):
        self._user = ''
        self._group = ''
        self._other = ''
        self._integer = 0
        self.user = user
        self.group = group
        self.other = other

    @classmethod
    def from_int(cls, i):
        i &= FULL_PERMS
        permissions = ('', 'x', 'w', 'xw', 'r', 'rx', 'rw', 'rwx')
        chunks = []
        while i:
            chunks.append(permissions[i & _SINGLE_FULL_PERM])
            i >>= 3
        chunks.reverse()
        return cls(*chunks)

    @classmethod
    def from_path(cls, path):
        return cls.from_int(stat.S_IMODE(os.stat(path).st_mode))

    def __int__(self):
        return self._integer

    user = _FilePermProperty('_user', 2)
    group = _FilePermProperty('_group', 1)
    other = _FilePermProperty('_other', 0)

    def __repr__(self):
        return '%s(user=%r, group=%r, other=%r)' % (
            self.__class__.__name__,
            self.user,
            self.group,
            self.other)


_TEXT_OPENFLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL
if hasattr(os, 'O_NOINHERIT'):
    _TEXT_OPENFLAGS |= os.O_NOINHERIT
if hasattr(os, 'O_NOFOLLOW'):
    _TEXT_OPENFLAGS |= os.O_NOFOLLOW

_BIN_OPENFLAGS = _TEXT_OPENFLAGS
if hasattr(os, 'O_BINARY'):
    _BIN_OPENFLAGS |= os.O_BINARY


try:
    import fcntl as fcntl
except ImportError:
    def set_cloexec(fd):
        pass
else:
    def set_cloexec(fd):
        try:
            current_flags = fcntl.fcntl(fd, fcntl.F_GETFD, 0)
        except OSError:
            pass
        else:
            current_flags |= fcntl.FD_CLOEXEC
            fcntl.fcntl(fd, fcntl.F_SETFD, current_flags)
        return


def path_to_unicode(path):
    if isinstance(path, str):
        return path
    encoding = sys.getfilesystemencoding() or sys.getdefaultencoding()
    return path.decode(encoding)


def atomic_save(dest_path, **kwargs):
    return AtomicSaver(dest_path, **kwargs)


class AtomicSaver:
    def __init__(self, dest_path, part_file=None, overwrite=True,
                 text_mode=False, keep_part=False, file_perms=None,
                 overwrite_part=False):
        self.dest_path = path_to_unicode(dest_path)
        if part_file is None:
            part_file = self.dest_path + '.part'
        self.part_path = path_to_unicode(part_file)
        self.overwrite = overwrite
        self.text_mode = text_mode
        self.keep_part = keep_part
        self.file_perms = file_perms
        self.overwrite_part = overwrite_part
        self.part_file = None

    def __enter__(self):
        permissions = self.file_perms
        if permissions is None:
            try:
                permissions = stat.S_IMODE(os.stat(self.dest_path).st_mode)
            except OSError:
                permissions = RW_PERMS

        flags = _TEXT_OPENFLAGS if self.text_mode else _BIN_OPENFLAGS
        if self.overwrite_part:
            flags = (flags & ~os.O_EXCL) | os.O_TRUNC

        fd = os.open(self.part_path, flags, permissions)
        set_cloexec(fd)
        mode = 'w+' if self.text_mode else 'w+b'
        self.part_file = os.fdopen(fd, mode)
        return self.part_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.part_file is not None:
            self.part_file.close()

        if exc_type is not None:
            if not self.keep_part:
                try:
                    os.unlink(self.part_path)
                except OSError:
                    pass
            return

        try:
            if self.overwrite:
                os.rename(self.part_path, self.dest_path)
            else:
                os.link(self.part_path, self.dest_path)
                os.unlink(self.part_path)
        except OSError:
            if not self.keep_part:
                try:
                    os.unlink(self.part_path)
                except OSError:
                    pass
            raise


def iter_find_files(directory, patterns, ignored=None, include_dirs=False):
    if isinstance(patterns, str):
        patterns = [patterns]
    if ignored is None:
        ignored = []
    elif isinstance(ignored, str):
        ignored = [ignored]

    pattern_matchers = [re.compile(fnmatch.translate(pattern))
                        for pattern in patterns]
    ignored_matchers = [re.compile(fnmatch.translate(pattern))
                       for pattern in ignored]

    for root, dirs, files in os.walk(directory):
        if include_dirs:
            files.extend(dirs)

        for basename in files:
            if not any(matcher.match(basename) for matcher in pattern_matchers):
                continue
            if any(matcher.match(basename) for matcher in ignored_matchers):
                continue
            yield os.path.join(root, basename)


def copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2,
             ignore_dangling_symlinks=False):
    entries = os.listdir(src)
    ignored_names = set() if ignore is None else ignore(src, entries)

    os.makedirs(dst)
    errors = []

    for name in entries:
        if name in ignored_names:
            continue

        source_name = os.path.join(src, name)
        destination_name = os.path.join(dst, name)

        try:
            if os.path.islink(source_name):
                link_target = os.readlink(source_name)
                if symlinks:
                    os.symlink(link_target, destination_name)
                    try:
                        copystat(source_name, destination_name,
                                 follow_symlinks=not symlinks)
                    except TypeError:
                        copystat(source_name, destination_name)
                else:
                    if (ignore_dangling_symlinks
                            and not os.path.exists(source_name)):
                        continue
                    if os.path.isdir(source_name):
                        copytree(source_name, destination_name, symlinks,
                                 ignore, copy_function,
                                 ignore_dangling_symlinks)
                    else:
                        copy_function(source_name, destination_name)
            elif os.path.isdir(source_name):
                copytree(source_name, destination_name, symlinks, ignore,
                         copy_function, ignore_dangling_symlinks)
            else:
                copy_function(source_name, destination_name)
        except Error as exc:
            errors.extend(exc.args[0])
        except OSError as exc:
            errors.append((source_name, destination_name, str(exc)))

    try:
        copystat(src, dst)
    except OSError as exc:
        if getattr(exc, 'winerror', None) is None:
            errors.append((src, dst, str(exc)))

    if errors:
        raise Error(errors)

    return dst