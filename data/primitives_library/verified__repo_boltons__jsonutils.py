import io
import os
import json


DEFAULT_BLOCKSIZE = 4096


__all__ = ['JSONLIterator', 'reverse_iter_lines']


def reverse_iter_lines(file_obj, blocksize=DEFAULT_BLOCKSIZE, preseek=True,
                       encoding=None):
    """Iterate over a seekable file's lines in reverse order."""
    encoding = encoding or getattr(file_obj, 'encoding', None)

    original_file_obj = file_obj
    try:
        file_obj = original_file_obj.buffer
    except (AttributeError, io.UnsupportedOperation):
        pass

    empty_bytes = b''
    newline_bytes = b'\n'
    empty_text = ''

    if preseek:
        file_obj.seek(0, os.SEEK_END)

    buffer = empty_bytes
    position = file_obj.tell()

    while position > 0:
        read_size = min(blocksize, position)
        position -= read_size
        file_obj.seek(position, os.SEEK_SET)
        current = file_obj.read(read_size)
        buffer = current + buffer
        lines = buffer.splitlines()

        if len(lines) < 2 or lines[0] == empty_bytes:
            continue

        if buffer[-1:] == newline_bytes:
            yield empty_text if encoding else empty_bytes

        for line in lines[:0:-1]:
            yield line.decode(encoding) if encoding else line

        buffer = lines[0]

    if buffer:
        yield buffer.decode(encoding) if encoding else buffer


class JSONLIterator:
    """Iterator for JSON Lines-formatted files."""

    def __init__(self, file_obj, ignore_errors=False, reverse=False,
                 rel_seek=None):
        self._reverse = bool(reverse)
        self._file_obj = file_obj
        self.ignore_errors = ignore_errors

        if rel_seek is None:
            if reverse:
                rel_seek = 1.0
        elif not -1.0 < rel_seek <= 1.0:
            raise ValueError("'rel_seek' expected a float between"
                             " -1.0 and 1.0, not %r" % rel_seek)
        elif rel_seek < 0:
            rel_seek = 1.0 + rel_seek

        self._rel_seek = rel_seek
        self._blocksize = DEFAULT_BLOCKSIZE

        if rel_seek is not None:
            self._init_rel_seek()

        if self._reverse:
            self._line_iter = reverse_iter_lines(
                self._file_obj,
                blocksize=self._blocksize,
                preseek=False,
            )
        else:
            self._line_iter = iter(self._file_obj)

    @property
    def cur_byte_pos(self):
        """Return the current position of the underlying file object."""
        return self._file_obj.tell()

    def _align_to_newline(self):
        """Move the current file position to the following newline."""
        file_obj = self._file_obj
        blocksize = self._blocksize
        newline = b'\n' if isinstance(file_obj.read(0), bytes) else '\n'
        current = newline[:0]
        total_read = 0
        current_position = file_obj.tell()

        while newline not in current:
            current = file_obj.read(blocksize)
            if not current:
                file_obj.seek(0, os.SEEK_END)
                return
            total_read += blocksize

        newline_offset = current.index(newline) + total_read - blocksize
        file_obj.seek(current_position + newline_offset)

    def _init_rel_seek(self):
        """Set the file position according to the configured relative seek."""
        rel_seek = self._rel_seek
        file_obj = self._file_obj

        if rel_seek == 0.0:
            file_obj.seek(0, os.SEEK_SET)
            return

        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell()

        if rel_seek == 1.0:
            self._cur_pos = size
            return

        target = int(size * rel_seek)
        file_obj.seek(target, os.SEEK_SET)
        self._align_to_newline()
        self._cur_pos = file_obj.tell()

    def __iter__(self):
        return self

    def next(self):
        """Return the next successfully decoded JSON object."""
        while True:
            line = next(self._line_iter).lstrip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                if not self.ignore_errors:
                    raise

    __next__ = next


if __name__ == '__main__':
    def _main():
        import sys

        if '-h' in sys.argv or '--help' in sys.argv:
            print('loads one or more JSON Line files for basic validation.')
            return

        verbose = '-v' in sys.argv or '--verbose' in sys.argv
        file_count = 0
        obj_count = 0

        for path in sys.argv[1:]:
            if path.startswith('-'):
                continue
            with open(path, 'rb') as file_obj:
                file_count += 1
                for obj in JSONLIterator(file_obj):
                    obj_count += 1
                    if verbose:
                        print(repr(obj))

        print('validated %s objects from %s files'
              % (obj_count, file_count))

    _main()