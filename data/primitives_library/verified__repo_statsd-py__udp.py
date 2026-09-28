import socket

from .base import PipelineBase, StatsClientBase


class Pipeline(PipelineBase):
    def __init__(self, client):
        super().__init__(client)
        self._maxudpsize = client._maxudpsize

    def _send(self):
        packet = self._stats.popleft()

        while self._stats:
            metric = self._stats.popleft()
            if len(packet) + len(metric) + 1 >= self._maxudpsize:
                self._client._after(packet)
                packet = metric
            else:
                packet = packet + '\n' + metric

        self._client._after(packet)


class StatsClient(StatsClientBase):
    """A client for statsd."""

    def __init__(self, host='localhost', port=8125, prefix=None,
                 maxudpsize=512, ipv6=False):
        family_hint = socket.AF_INET6 if ipv6 else socket.AF_INET
        family, _, _, _, address = socket.getaddrinfo(
            host, port, family_hint, socket.SOCK_DGRAM
        )[0]

        self._addr = address
        self._sock = socket.socket(family, socket.SOCK_DGRAM)
        self._prefix = prefix
        self._maxudpsize = maxudpsize

    def _send(self, data):
        """Send data to statsd."""
        try:
            self._sock.sendto(data.encode('ascii'), self._addr)
        except (OSError, RuntimeError):
            pass

    def close(self):
        if self._sock and hasattr(self._sock, 'close'):
            self._sock.close()
        self._sock = None

    def pipeline(self):
        return Pipeline(self)