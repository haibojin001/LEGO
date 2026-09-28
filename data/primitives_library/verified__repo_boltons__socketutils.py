import socket
import time

try:
    from threading import RLock
except Exception:
    class RLock:
        def __enter__(self):
            return self

        def __exit__(self, exctype, excinst, exctb):
            return False

try:
    from .typeutils import make_sentinel
    _UNSET = make_sentinel(var_name='_UNSET')
except ImportError:
    _UNSET = object()


DEFAULT_TIMEOUT = 10
DEFAULT_MAXSIZE = 32 * 1024
_RECV_LARGE_MAXSIZE = 1024 ** 5


class Timeout(socket.timeout):
    """Raised when a BufferedSocket operation exceeds its timeout."""


class ConnectionClosed(socket.error):
    """Raised when a peer closes a connection before an operation completes."""


class MessageTooLong(socket.error):
    """Raised when an incoming message exceeds its configured maximum size."""


class NetstringMessageTooLong(MessageTooLong):
    """Raised when a netstring payload exceeds its configured maximum size."""


class NetstringProtocolError(socket.error):
    """Raised when invalid netstring framing is received."""


class BufferedSocket:
    def __init__(self, sock, timeout=_UNSET,
                 maxsize=DEFAULT_MAXSIZE, recvsize=_UNSET):
        self.sock = sock
        self.rbuf = b''
        self.sbuf = []
        self.maxsize = int(maxsize)

        if timeout is _UNSET:
            sock_timeout = self.sock.gettimeout()
            self.timeout = DEFAULT_TIMEOUT if sock_timeout is None else sock_timeout
        elif timeout is None:
            self.timeout = None
        else:
            self.timeout = float(timeout)

        self._recvsize = self.maxsize if recvsize is _UNSET else int(recvsize)
        self._send_lock = RLock()
        self._recv_lock = RLock()

    def settimeout(self, timeout):
        self.timeout = timeout

    def gettimeout(self):
        return self.timeout

    def setblocking(self, blocking):
        self.timeout = None if blocking else 0.0

    def setmaxsize(self, maxsize):
        self.maxsize = maxsize

    def getrecvbuffer(self):
        with self._recv_lock:
            return self.rbuf

    def getsendbuffer(self):
        with self._send_lock:
            return b''.join(self.sbuf)

    def _make_deadline(self, timeout):
        if timeout is None or timeout == 0:
            return None
        return time.time() + timeout

    def _remaining_timeout(self, timeout, deadline):
        if deadline is None:
            return timeout
        remaining = deadline - time.time()
        if remaining <= 0:
            raise Timeout('timed out')
        return remaining

    def _call_socket(self, method, timeout, deadline, *args):
        old_timeout = self.sock.gettimeout()
        current_timeout = self._remaining_timeout(timeout, deadline)
        changed = old_timeout != current_timeout
        if changed:
            self.sock.settimeout(current_timeout)
        try:
            return method(*args)
        except socket.timeout as exc:
            raise Timeout(str(exc) or 'timed out')
        finally:
            if changed:
                try:
                    self.sock.settimeout(old_timeout)
                except Exception:
                    pass

    def _recv_once(self, size, timeout, deadline):
        return self._call_socket(self.sock.recv, timeout, deadline, size)

    def _send_once(self, data, timeout, deadline):
        return self._call_socket(self.sock.send, timeout, deadline, data)

    def recv(self, size, flags=0, timeout=_UNSET):
        with self._recv_lock:
            if flags:
                raise ValueError('non-zero flags not supported: %r' % flags)
            if timeout is _UNSET:
                timeout = self.timeout

            if size == 0:
                return b''

            if len(self.rbuf) >= size:
                ret, self.rbuf = self.rbuf[:size], self.rbuf[size:]
                return ret

            buffered = self.rbuf
            self.rbuf = b''
            deadline = self._make_deadline(timeout)
            data = self._recv_once(size - len(buffered), timeout, deadline)
            return buffered + data

    def recv_until(self, delimiter, timeout=_UNSET, maxsize=_UNSET):
        with self._recv_lock:
            if not delimiter:
                raise ValueError('delimiter must not be empty')
            if timeout is _UNSET:
                timeout = self.timeout
            if maxsize is _UNSET:
                maxsize = self.maxsize

            deadline = self._make_deadline(timeout)
            while True:
                index = self.rbuf.find(delimiter)
                if index >= 0:
                    if index > maxsize:
                        raise MessageTooLong(
                            'message exceeded maximum size of %d bytes' % maxsize)
                    ret = self.rbuf[:index]
                    self.rbuf = self.rbuf[index + len(delimiter):]
                    return ret

                if len(self.rbuf) > maxsize:
                    raise MessageTooLong(
                        'message exceeded maximum size of %d bytes' % maxsize)

                data = self._recv_once(self._recvsize, timeout, deadline)
                if not data:
                    raise ConnectionClosed('connection closed while receiving data')
                self.rbuf += data

    def recv_size(self, size, timeout=_UNSET):
        with self._recv_lock:
            if timeout is _UNSET:
                timeout = self.timeout

            if size > self.maxsize:
                raise MessageTooLong(
                    'message size %d exceeds maximum size of %d bytes'
                    % (size, self.maxsize))
            if size <= 0:
                return b''

            deadline = self._make_deadline(timeout)
            while len(self.rbuf) < size:
                want = max(self._recvsize, size - len(self.rbuf))
                data = self._recv_once(want, timeout, deadline)
                if not data:
                    raise ConnectionClosed('connection closed while receiving data')
                self.rbuf += data

            ret, self.rbuf = self.rbuf[:size], self.rbuf[size:]
            return ret

    def recv_close(self, timeout=_UNSET, maxsize=_UNSET):
        with self._recv_lock:
            if timeout is _UNSET:
                timeout = self.timeout
            if maxsize is _UNSET:
                maxsize = self.maxsize

            deadline = self._make_deadline(timeout)
            while True:
                if len(self.rbuf) > maxsize:
                    raise MessageTooLong(
                        'message exceeded maximum size of %d bytes' % maxsize)

                data = self._recv_once(self._recvsize, timeout, deadline)
                if not data:
                    ret, self.rbuf = self.rbuf, b''
                    return ret
                self.rbuf += data

    def peek(self, nbytes, flags=0):
        with self._recv_lock:
            if flags:
                raise ValueError('non-zero flags not supported: %r' % flags)
            if nbytes <= 0:
                return b''
            if len(self.rbuf) >= nbytes:
                return self.rbuf[:nbytes]

            buffered = self.rbuf
            self.rbuf = b''
            try:
                data = self.recv(nbytes - len(buffered), timeout=self.timeout)
            finally:
                self.rbuf = buffered + self.rbuf
            self.rbuf = buffered + data + self.rbuf[len(buffered):]
            return self.rbuf[:nbytes]

    def buffer(self, data):
        with self._send_lock:
            if data:
                self.sbuf.append(data)

    def send(self, data, flags=0, timeout=_UNSET):
        with self._send_lock:
            if flags:
                raise ValueError('non-zero flags not supported: %r' % flags)
            if timeout is _UNSET:
                timeout = self.timeout
            if not data:
                return 0

            self.sbuf.append(data)
            deadline = self._make_deadline(timeout)
            pending = b''.join(self.sbuf)
            sent = self._send_once(pending, timeout, deadline)
            if sent == 0:
                raise ConnectionClosed('connection closed while sending data')
            self._consume_sendbuffer(sent)
            return sent

    def _consume_sendbuffer(self, count):
        while count and self.sbuf:
            part = self.sbuf[0]
            if count >= len(part):
                count -= len(part)
                del self.sbuf[0]
            else:
                self.sbuf[0] = part[count:]
                count = 0

    def flush(self, timeout=_UNSET):
        with self._send_lock:
            if timeout is _UNSET:
                timeout = self.timeout
            deadline = self._make_deadline(timeout)
            while self.sbuf:
                pending = b''.join(self.sbuf)
                sent = self._send_once(pending, timeout, deadline)
                if sent == 0:
                    raise ConnectionClosed('connection closed while sending data')
                self._consume_sendbuffer(sent)

    def sendall(self, data, flags=0, timeout=_UNSET):
        with self._send_lock:
            if flags:
                raise ValueError('non-zero flags not supported: %r' % flags)
            if timeout is _UNSET:
                timeout = self.timeout
            if data:
                self.sbuf.append(data)
            self.flush(timeout=timeout)


class NetstringSocket:
    def __init__(self, sock, timeout=_UNSET, maxsize=DEFAULT_MAXSIZE):
        self.sock = sock
        self.timeout = DEFAULT_TIMEOUT if timeout is _UNSET else timeout
        self.maxsize = int(maxsize)
        self.bsock = BufferedSocket(sock, timeout=timeout,
                                    maxsize=_RECV_LARGE_MAXSIZE)

    def settimeout(self, timeout):
        self.timeout = timeout
        self.bsock.settimeout(timeout)

    def gettimeout(self):
        return self.bsock.gettimeout()

    def setblocking(self, blocking):
        self.timeout = None if blocking else 0.0
        self.bsock.setblocking(blocking)

    def setmaxsize(self, maxsize):
        self.maxsize = maxsize

    def read_ns(self, timeout=_UNSET, maxsize=_UNSET):
        if timeout is _UNSET:
            timeout = self.bsock.gettimeout()
        if maxsize is _UNSET:
            maxsize = self.maxsize

        try:
            size_text = self.bsock.recv_until(
                b':', timeout=timeout, maxsize=_RECV_LARGE_MAXSIZE)
        except MessageTooLong:
            raise NetstringProtocolError('netstring length declaration is too long')

        if (not size_text or
                any(ch < 48 or ch > 57 for ch in size_text) or
                (len(size_text) > 1 and size_text[:1] == b'0')):
            raise NetstringProtocolError(
                'invalid netstring length declaration: %r' % size_text)

        try:
            size = int(size_text)
        except ValueError:
            raise NetstringProtocolError(
                'invalid netstring length declaration: %r' % size_text)

        if size > maxsize:
            raise NetstringMessageTooLong(
                'netstring payload size %d exceeds maximum size of %d bytes'
                % (size, maxsize))

        payload = self.bsock.recv_size(size, timeout=timeout)
        terminator = self.bsock.recv(1, timeout=timeout)
        if terminator != b',':
            raise NetstringProtocolError(
                'expected trailing comma, got %r' % terminator)
        return payload

    def write_ns(self, payload, timeout=_UNSET):
        if timeout is _UNSET:
            timeout = self.bsock.gettimeout()
        if len(payload) > self.maxsize:
            raise NetstringMessageTooLong(
                'netstring payload size %d exceeds maximum size of %d bytes'
                % (len(payload), self.maxsize))
        message = str(len(payload)).encode('ascii') + b':' + payload + b','
        return self.bsock.sendall(message, timeout=timeout)