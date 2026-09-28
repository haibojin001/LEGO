import errno
import selectors
import socket
from typing import Any, Optional, Union

from ._exceptions import (
    WebSocketConnectionClosedException,
    WebSocketTimeoutException,
)
from ._ssl_compat import SSLError, SSLEOFError, SSLWantReadError, SSLWantWriteError
from ._utils import extract_error_code, extract_err_message


DEFAULT_SOCKET_OPTION = [(socket.SOL_TCP, socket.TCP_NODELAY, 1)]

if hasattr(socket, "SO_KEEPALIVE"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
if hasattr(socket, "TCP_KEEPIDLE"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_TCP, socket.TCP_KEEPIDLE, 30))
if hasattr(socket, "TCP_KEEPINTVL"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_TCP, socket.TCP_KEEPINTVL, 10))
if hasattr(socket, "TCP_KEEPCNT"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_TCP, socket.TCP_KEEPCNT, 3))


_default_timeout = None


__all__ = [
    "DEFAULT_SOCKET_OPTION",
    "sock_opt",
    "setdefaulttimeout",
    "getdefaulttimeout",
    "recv",
    "recv_line",
    "send",
]


class sock_opt:
    def __init__(
        self,
        sockopt: Optional[list[tuple]],
        sslopt: Optional[dict[str, Any]],
    ) -> None:
        self.sockopt = [] if sockopt is None else sockopt
        self.sslopt = {} if sslopt is None else sslopt
        self.timeout: Optional[Union[int, float]] = None


def setdefaulttimeout(timeout: Optional[Union[int, float]]) -> None:
    global _default_timeout
    _default_timeout = timeout


def getdefaulttimeout() -> Optional[Union[int, float]]:
    return _default_timeout


def recv(sock: socket.socket, bufsize: int) -> bytes:
    if not sock:
        raise WebSocketConnectionClosedException("socket is already closed.")

    def receive_with_wait() -> bytes:
        try:
            return sock.recv(bufsize)
        except SSLWantReadError:
            pass
        except socket.error as exc:
            code = extract_error_code(exc)
            if code not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise

        selector = selectors.DefaultSelector()
        try:
            selector.register(sock, selectors.EVENT_READ)
            available = selector.select(sock.gettimeout())
        finally:
            selector.close()

        if not available:
            raise WebSocketTimeoutException("Connection timed out waiting for data")

        return sock.recv(bufsize)

    try:
        if sock.gettimeout() == 0:
            result = sock.recv(bufsize)
        else:
            result = receive_with_wait()
    except TimeoutError:
        raise WebSocketTimeoutException("Connection timed out")
    except socket.timeout as exc:
        raise WebSocketTimeoutException(extract_err_message(exc))
    except SSLError as exc:
        message = extract_err_message(exc)
        if isinstance(message, str) and "timed out" in message:
            raise WebSocketTimeoutException(message)
        raise

    if result is None or not result:
        raise WebSocketConnectionClosedException("Connection to remote host was lost.")

    return result


def recv_line(sock: socket.socket) -> bytes:
    pieces = []
    while True:
        piece = recv(sock, 1)
        pieces.append(piece)
        if piece == b"\n":
            return b"".join(pieces)


def send(sock: socket.socket, data: Union[bytes, str]) -> int:
    if isinstance(data, str):
        data = data.encode("utf-8")

    if not sock:
        raise WebSocketConnectionClosedException("socket is already closed.")

    def send_with_wait() -> int:
        try:
            return sock.send(data)
        except SSLEOFError:
            raise WebSocketConnectionClosedException("socket is already closed.")
        except SSLWantWriteError:
            pass
        except socket.error as exc:
            code = extract_error_code(exc)
            if code is None or code not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise

        selector = selectors.DefaultSelector()
        try:
            selector.register(sock, selectors.EVENT_WRITE)
            available = selector.select(sock.gettimeout())
        finally:
            selector.close()

        if not available:
            return 0

        return sock.send(data)

    try:
        if sock.gettimeout() == 0:
            return sock.send(data)
        return send_with_wait()
    except socket.timeout as exc:
        raise WebSocketTimeoutException(extract_err_message(exc))
    except (OSError, SSLError) as exc:
        message = extract_err_message(exc)
        if isinstance(message, str) and "timed out" in message:
            raise WebSocketTimeoutException(message)
        raise