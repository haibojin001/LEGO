import socket

from .base import PipelineBase, StatsClientBase


class StreamPipeline(PipelineBase):
    def _send(self):
        payload = '\n'.join(self._stats)
        self._client._after(payload)
        self._stats.clear()


class StreamClientBase(StatsClientBase):
    def connect(self):
        raise NotImplementedError()

    def close(self):
        sock = self._sock
        if sock and hasattr(sock, 'close'):
            sock.close()
        self._sock = None

    def reconnect(self):
        self.close()
        self.connect()

    def pipeline(self):
        return StreamPipeline(self)

    def _send(self, data):
        if not self._sock:
            self.connect()
        self._do_send(data)

    def _do_send(self, data):
        payload = data.encode('ascii') + b'\n'
        self._sock.sendall(payload)


class TCPStatsClient(StreamClientBase):
    """TCP version of StatsClient."""

    def __init__(self, host='localhost', port=8125, prefix=None,
                 timeout=None, ipv6=False):
        """Create a new client."""
        self._host = host
        self._port = port
        self._ipv6 = ipv6
        self._timeout = timeout
        self._prefix = prefix
        self._sock = None

    def connect(self):
        requested_family = socket.AF_INET6 if self._ipv6 else socket.AF_INET
        result = socket.getaddrinfo(
            self._host,
            self._port,
            requested_family,
            socket.SOCK_STREAM,
        )[0]
        family = result[0]
        address = result[4]
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect(address)


class UnixSocketStatsClient(StreamClientBase):
    """Unix domain socket version of StatsClient."""

    def __init__(self, socket_path, prefix=None, timeout=None):
        """Create a new client."""
        self._socket_path = socket_path
        self._timeout = timeout
        self._prefix = prefix
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect(self._socket_path)