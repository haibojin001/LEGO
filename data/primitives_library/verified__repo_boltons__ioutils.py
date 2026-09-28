import os
from io import BytesIO, StringIO, IOBase
from abc import ABCMeta, abstractmethod, abstractproperty
from errno import EINVAL
from codecs import EncodedFile
from tempfile import TemporaryFile
from itertools import zip_longest

READ_CHUNK_SIZE = 21333


class SpooledIOBase(IOBase, metaclass=ABCMeta):
    def __init__(self, max_size=5000000, dir=None):
        self._max_size = max_size
        self._dir = dir

    def _checkClosed(self, msg=None):
        if self.closed:
            raise ValueError('I/O operation on closed file.' if msg is None else msg)

    @abstractmethod
    def read(self, n=-1):
        raise NotImplementedError

    @abstractmethod
    def write(self, s):
        raise NotImplementedError

    @abstractmethod
    def seek(self, pos, mode=0):
        raise NotImplementedError

    @abstractmethod
    def readline(self, length=None):
        raise NotImplementedError

    @abstractmethod
    def readlines(self, sizehint=0):
        raise NotImplementedError

    def writelines(self, lines):
        self._checkClosed()
        for line in lines:
            self.write(line)

    @abstractmethod
    def rollover(self):
        raise NotImplementedError

    @abstractmethod
    def tell(self):
        raise NotImplementedError

    @abstractproperty
    def buffer(self):
        raise NotImplementedError

    @abstractproperty
    def _rolled(self):
        raise NotImplementedError

    @abstractproperty
    def len(self):
        raise NotImplementedError

    def _get_softspace(self):
        return self.buffer.softspace

    def _set_softspace(self, val):
        self.buffer.softspace = val

    softspace = property(_get_softspace, _set_softspace)

    @property
    def _file(self):
        return self.buffer

    def close(self):
        return self.buffer.close()

    def flush(self):
        self._checkClosed()
        return self.buffer.flush()

    def isatty(self):
        self._checkClosed()
        return self.buffer.isatty()

    @property
    def closed(self):
        return self.buffer.closed

    @property
    def pos(self):
        return self.tell()

    @property
    def buf(self):
        return self.getvalue()

    def fileno(self):
        self.rollover()
        return self.buffer.fileno()

    def truncate(self, size=None):
        self._checkClosed()
        if size is None:
            return self.buffer.truncate()

        if size < 0:
            raise OSError(EINVAL, 'Negative size not allowed')

        old_pos = self.tell()
        self.seek(size)
        self.buffer.truncate()
        if old_pos < size:
            self.seek(old_pos)

    def getvalue(self):
        self._checkClosed()
        old_pos = self.tell()
        self.seek(0)
        ret = self.read()
        self.seek(old_pos)
        return ret

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return True

    def __next__(self):
        self._checkClosed()
        line = self.readline()
        if not line:
            pos = self.buffer.tell()
            self.buffer.seek(0, os.SEEK_END)
            if pos == self.buffer.tell():
                raise StopIteration
            self.buffer.seek(pos)
        return line

    next = __next__

    def __len__(self):
        return self.len

    def __iter__(self):
        self._checkClosed()
        return self

    def __enter__(self):
        self._checkClosed()
        return self

    def __exit__(self, *args):
        self._file.close()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        self_pos = self.tell()
        other_pos = other.tell()
        try:
            self.seek(0)
            other.seek(0)
            result = True
            for left, right in zip_longest(self, other):
                if left != right:
                    result = False
                    break
            self.seek(self_pos)
            other.seek(other_pos)
        except Exception:
            try:
                self.seek(self_pos)
            except Exception:
                pass
            try:
                other.seek(other_pos)
            except Exception:
                pass
            raise
        return result

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SpooledBytesIO(SpooledIOBase):
    def __init__(self, max_size=5000000, dir=None):
        super().__init__(max_size=max_size, dir=dir)
        self._buffer = BytesIO()
        self.__rolled = False

    @property
    def buffer(self):
        return self._buffer

    @property
    def _rolled(self):
        return self.__rolled

    @property
    def len(self):
        self._checkClosed()
        old_pos = self.tell()
        self.seek(0, os.SEEK_END)
        length = self.tell()
        self.seek(old_pos)
        return length

    def read(self, n=-1):
        self._checkClosed()
        return self.buffer.read(n)

    def readinto(self, b):
        self._checkClosed()
        return self.buffer.readinto(b)

    def write(self, s):
        self._checkClosed()
        if not isinstance(s, bytes):
            raise TypeError('bytes expected, got {}'.format(type(s).__name__))
        if self.tell() + len(s) >= self._max_size:
            self.rollover()
        self.buffer.write(s)

    def seek(self, pos, mode=0):
        self._checkClosed()
        return self.buffer.seek(pos, mode)

    def tell(self):
        self._checkClosed()
        return self.buffer.tell()

    def readline(self, length=None):
        self._checkClosed()
        if length is None:
            return self.buffer.readline()
        return self.buffer.readline(length)

    def readlines(self, sizehint=0):
        self._checkClosed()
        return self.buffer.readlines(sizehint)

    def rollover(self):
        self._checkClosed()
        if self.__rolled:
            return

        old_buffer = self._buffer
        old_pos = old_buffer.tell()
        new_buffer = TemporaryFile(mode='w+b', dir=self._dir)
        old_buffer.seek(0)
        while True:
            chunk = old_buffer.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            new_buffer.write(chunk)
        new_buffer.seek(old_pos)
        self._buffer = new_buffer
        self.__rolled = True
        old_buffer.close()


class SpooledStringIO(SpooledIOBase):
    def __init__(self, max_size=5000000, dir=None, encoding='utf-8',
                 errors='strict'):
        super().__init__(max_size=max_size, dir=dir)
        self.encoding = encoding
        self.errors = errors
        self._buffer = StringIO()
        self.__rolled = False

    @property
    def buffer(self):
        return self._buffer

    @property
    def _rolled(self):
        return self.__rolled

    @property
    def len(self):
        self._checkClosed()
        old_pos = self.tell()
        self.seek(0, os.SEEK_END)
        length = self.tell()
        self.seek(old_pos)
        return length

    def read(self, n=-1):
        self._checkClosed()
        return self.buffer.read(n)

    def write(self, s):
        self._checkClosed()
        if not isinstance(s, str):
            raise TypeError('str expected, got {}'.format(type(s).__name__))
        if self.tell() + len(s) >= self._max_size:
            self.rollover()
        self.buffer.write(s)

    def seek(self, pos, mode=0):
        self._checkClosed()
        return self.buffer.seek(pos, mode)

    def tell(self):
        self._checkClosed()
        return self.buffer.tell()

    def readline(self, length=None):
        self._checkClosed()
        if length is None:
            return self.buffer.readline()
        return self.buffer.readline(length)

    def readlines(self, sizehint=0):
        self._checkClosed()
        return self.buffer.readlines(sizehint)

    def rollover(self):
        self._checkClosed()
        if self.__rolled:
            return

        old_buffer = self._buffer
        old_pos = old_buffer.tell()
        new_buffer = TemporaryFile(mode='w+t',
                                   encoding=self.encoding,
                                   errors=self.errors,
                                   dir=self._dir)
        old_buffer.seek(0)
        while True:
            chunk = old_buffer.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            new_buffer.write(chunk)
        new_buffer.seek(old_pos)
        self._buffer = new_buffer
        self.__rolled = True
        old_buffer.close()


class MultiFileReader(IOBase):
    def __init__(self, *fileobjs):
        self._fileobjs = list(fileobjs)
        self._cur_file_idx = 0
        self._pos = 0
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def _check_closed(self):
        if self._closed:
            raise ValueError('I/O operation on closed file.')

    @property
    def _cur_file(self):
        if self._cur_file_idx >= len(self._fileobjs):
            return None
        return self._fileobjs[self._cur_file_idx]

    def _advance(self):
        self._cur_file_idx += 1

    def _empty_value(self):
        current = self._cur_file
        if current is not None:
            try:
                value = current.read(0)
                return value
            except Exception:
                pass
        for fileobj in self._fileobjs:
            try:
                return fileobj.read(0)
            except Exception:
                continue
        return b''

    def read(self, n=-1):
        self._check_closed()
        if n is None or n < 0:
            pieces = []
            while self._cur_file is not None:
                piece = self._cur_file.read()
                if piece:
                    pieces.append(piece)
                    self._pos += len(piece)
                self._advance()
            if not pieces:
                return self._empty_value()
            return pieces[0][:0].join(pieces)

        if n == 0:
            return self._empty_value()

        pieces = []
        remaining = n
        while remaining and self._cur_file is not None:
            piece = self._cur_file.read(remaining)
            if piece:
                pieces.append(piece)
                self._pos += len(piece)
                remaining -= len(piece)
            if not piece or remaining:
                self._advance()

        if not pieces:
            return self._empty_value()
        return pieces[0][:0].join(pieces)

    def readline(self, size=-1):
        self._check_closed()
        if size is None:
            size = -1
        if size == 0:
            return self._empty_value()

        pieces = []
        remaining = size
        while self._cur_file is not None and remaining != 0:
            if remaining < 0:
                piece = self._cur_file.readline()
            else:
                piece = self._cur_file.readline(remaining)

            if piece:
                pieces.append(piece)
                self._pos += len(piece)
                if remaining > 0:
                    remaining -= len(piece)
                if piece.endswith(b'\n') or piece.endswith('\n'):
                    break

            if not piece or (piece and not (piece.endswith(b'\n') or piece.endswith('\n'))):
                self._advance()

        if not pieces:
            return self._empty_value()
        return pieces[0][:0].join(pieces)

    def readlines(self, hint=-1):
        self._check_closed()
        lines = []
        total = 0
        while hint < 0 or total < hint:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            total += len(line)
        return lines

    def tell(self):
        self._check_closed()
        return self._pos

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    def flush(self):
        self._check_closed()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for fileobj in self._fileobjs:
            try:
                fileobj.close()
            except Exception:
                pass

    def __iter__(self):
        self._check_closed()
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    next = __next__