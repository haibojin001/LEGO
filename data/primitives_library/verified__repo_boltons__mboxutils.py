import mailbox
import tempfile


DEFAULT_MAXMEM = 4 * 1024 * 1024


class mbox_readonlydir(mailbox.mbox):
    """An mbox mailbox which rewrites changed contents in place."""

    def __init__(self, path, factory=None, create=True, maxmem=1024 * 1024):
        super().__init__(path, factory, create)
        self.maxmem = maxmem

    def flush(self):
        """Write pending mailbox changes without replacing the mailbox file."""
        if not self._pending:
            if self._pending_sync:
                mailbox._sync_flush(self._file)
                self._pending_sync = False
            return

        assert self._toc is not None

        self._file.seek(0, 2)
        current_length = self._file.tell()
        if current_length != self._file_length:
            raise mailbox.ExternalClashError(
                'Size of mailbox file changed (expected %i, found %i)'
                % (self._file_length, current_length)
            )

        self._file.seek(0)

        with tempfile.TemporaryFile() as temporary:
            updated_toc = {}
            self._pre_mailbox_hook(temporary)

            for key in sorted(self._toc):
                start, stop = self._toc[key]
                self._file.seek(start)

                self._pre_message_hook(temporary)
                new_start = temporary.tell()

                while self._file.tell() < stop:
                    amount = min(4096, stop - self._file.tell())
                    data = self._file.read(amount)
                    if not data:
                        break
                    temporary.write(data)

                updated_toc[key] = (new_start, temporary.tell())
                self._post_message_hook(temporary)

            self._file_length = temporary.tell()
            self._file.seek(0)
            temporary.seek(0)

            if self._file_length <= self.maxmem:
                self._file.write(temporary.read())
            else:
                while True:
                    data = temporary.read(4096)
                    if not data:
                        break
                    self._file.write(data)

            self._file.truncate()

        self._toc = updated_toc
        self._pending = False
        self._pending_sync = False

        if self._locked:
            mailbox._lock_file(self._file, dotlock=False)