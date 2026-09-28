import inspect
import selectors
import socket
import time
from typing import Any, TYPE_CHECKING, Callable, Optional, Union

from ._logging import info
from ._socket import send

if TYPE_CHECKING:
    from ._app import WebSocketApp


class DispatcherBase:
    """
    Base implementation for websocket event dispatchers.
    """

    def __init__(
        self, app: "WebSocketApp", ping_timeout: Optional[Union[float, int]]
    ) -> None:
        self.app = app
        self.ping_timeout = ping_timeout

    def timeout(self, seconds: Optional[Union[float, int]], callback: Callable) -> None:
        if seconds is not None:
            time.sleep(seconds)
        callback()

    def reconnect(self, seconds: int, reconnector: Callable) -> None:
        try:
            depth = len(inspect.stack())
            info(f"reconnect() - retrying in {seconds} seconds [{depth} frames in stack]")
            time.sleep(seconds)
            reconnector(reconnecting=True)
        except KeyboardInterrupt as error:
            info(f"User exited {error}")
            raise error

    def send(self, sock: socket.socket, data: Union[str, bytes]) -> int:
        return send(sock, data)


class Dispatcher(DispatcherBase):
    """
    Selector-backed dispatcher for ordinary sockets.
    """

    def read(
        self,
        sock: socket.socket,
        read_callback: Callable,
        check_callback: Callable,
    ) -> None:
        if self.app.sock is None or self.app.sock.sock is None:
            return

        selector = selectors.DefaultSelector()
        selector.register(self.app.sock.sock, selectors.EVENT_READ)

        try:
            while self.app.keep_running:
                if selector.select(self.ping_timeout):
                    if not read_callback():
                        break
                check_callback()
        finally:
            selector.close()


class SSLDispatcher(DispatcherBase):
    """
    Selector-backed dispatcher which also accounts for SSL buffered input.
    """

    def read(
        self,
        sock: socket.socket,
        read_callback: Callable,
        check_callback: Callable,
    ) -> None:
        if self.app.sock is None or self.app.sock.sock is None:
            return

        sock = self.app.sock.sock
        selector = selectors.DefaultSelector()
        selector.register(sock, selectors.EVENT_READ)

        try:
            while self.app.keep_running:
                if self.select(sock, selector):
                    if not read_callback():
                        break
                check_callback()
        finally:
            selector.close()

    def select(self, sock: Any, sel: selectors.DefaultSelector) -> Any:
        if self.app.sock is None:
            return None

        sock = self.app.sock.sock
        if sock.pending():
            return [sock]

        events = sel.select(self.ping_timeout)
        if len(events) > 0:
            return events[0][0]
        return None


class WrappedDispatcher:
    """
    Adapter for externally supplied dispatcher implementations.
    """

    def __init__(
        self,
        app: "WebSocketApp",
        ping_timeout: Optional[Union[float, int]],
        dispatcher: Any,
        handleDisconnect: Optional[Callable],
    ) -> None:
        self.app = app
        self.ping_timeout = ping_timeout
        self.dispatcher = dispatcher
        self.handleDisconnect = handleDisconnect
        dispatcher.signal(2, dispatcher.abort)

    def read(
        self,
        sock: socket.socket,
        read_callback: Callable,
        check_callback: Callable,
    ) -> None:
        self.dispatcher.read(sock, read_callback)
        if self.ping_timeout:
            self.timeout(self.ping_timeout, check_callback)

    def send(self, sock: socket.socket, data: Union[str, bytes]) -> int:
        self.dispatcher.buffwrite(sock, data, send, self.handleDisconnect)
        return len(data)

    def timeout(self, seconds: float, callback: Callable, *args: Any) -> None:
        self.dispatcher.timeout(seconds, callback, *args)

    def reconnect(self, seconds: int, reconnector: Callable) -> None:
        self.timeout(seconds, reconnector, True)