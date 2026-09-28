import inspect
import socket
import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Union

from ._logging import debug, error, info, warning
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import (
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketTimeoutException,
)
from ._ssl_compat import SSLError
from ._url import parse_url
from ._dispatcher import Dispatcher, DispatcherBase, SSLDispatcher, WrappedDispatcher


__all__ = ["WebSocketApp"]


RECONNECT = 0


def set_reconnect(reconnectInterval: int) -> None:
    global RECONNECT
    RECONNECT = reconnectInterval


class WebSocketApp:
    """
    A high-level WebSocket client interface modeled after the JavaScript
    WebSocket object.
    """

    def __init__(
        self,
        url: str,
        header: Optional[
            Union[
                list[str],
                dict[str, str],
                Callable[[], Union[list[str], dict[str, str]]],
            ]
        ] = None,
        on_open: Optional[Callable[["WebSocketApp"], None]] = None,
        on_reconnect: Optional[Callable[["WebSocketApp"], None]] = None,
        on_message: Optional[Callable[["WebSocketApp", Any], None]] = None,
        on_error: Optional[Callable[["WebSocketApp", Any], None]] = None,
        on_close: Optional[Callable[["WebSocketApp", Any, Any], None]] = None,
        on_ping: Optional[Callable] = None,
        on_pong: Optional[Callable] = None,
        on_cont_message: Optional[Callable] = None,
        keep_running: bool = True,
        get_mask_key: Optional[Callable] = None,
        cookie: Optional[str] = None,
        subprotocols: Optional[list[str]] = None,
        on_data: Optional[Callable] = None,
        socket: Optional[socket.socket] = None,
    ) -> None:
        self.url = url
        self.header = header if header is not None else []
        self.cookie = cookie

        self.on_open = on_open
        self.on_reconnect = on_reconnect
        self.on_message = on_message
        self.on_data = on_data
        self.on_error = on_error
        self.on_close = on_close
        self.on_ping = on_ping
        self.on_pong = on_pong
        self.on_cont_message = on_cont_message

        self.keep_running = False
        self.get_mask_key = get_mask_key
        self.sock: Optional[WebSocket] = None

        self.last_ping_tm = float(0)
        self.last_pong_tm = float(0)
        self.ping_thread: Optional[threading.Thread] = None
        self.stop_ping: Optional[threading.Event] = None
        self.ping_interval = float(0)
        self.ping_timeout: Optional[Union[float, int]] = None
        self.ping_payload = ""

        self.subprotocols = subprotocols
        self.prepared_socket = socket
        self.has_errored = False
        self.has_done_teardown = False
        self.has_done_teardown_lock = threading.Lock()
        self.last_close_frame: Optional[ABNF] = None

    def send(self, data: Union[bytes, str], opcode: int = ABNF.OPCODE_TEXT) -> None:
        if not self.sock or self.sock.send(data, opcode) == 0:
            raise WebSocketConnectionClosedException("Connection is already closed.")

    def send_text(self, text_data: str) -> None:
        if not self.sock or self.sock.send(text_data, ABNF.OPCODE_TEXT) == 0:
            raise WebSocketConnectionClosedException("Connection is already closed.")

    def send_bytes(self, data: Union[bytes, bytearray]) -> None:
        if not self.sock or self.sock.send(data, ABNF.OPCODE_BINARY) == 0:
            raise WebSocketConnectionClosedException("Connection is already closed.")

    def close(self, **kwargs: Any) -> None:
        self.keep_running = False
        sock = self.sock
        if sock:
            sock.close(**kwargs)
            if sock.close_frame is not None:
                self.last_close_frame = sock.close_frame
            self.sock = None

    def _start_ping_thread(self) -> None:
        self.last_ping_tm = float(0)
        self.last_pong_tm = float(0)
        self.stop_ping = threading.Event()
        self.ping_thread = threading.Thread(target=self._send_ping)
        self.ping_thread.daemon = True
        self.ping_thread.start()

    def _stop_ping_thread(self) -> None:
        stop_event = self.stop_ping
        ping_thread = self.ping_thread

        if stop_event is not None:
            stop_event.set()

        if (
            ping_thread is not None
            and ping_thread.is_alive()
            and ping_thread is not threading.current_thread()
        ):
            ping_thread.join(3)
            if ping_thread.is_alive():
                warning(
                    "Ping thread failed to terminate within 3 seconds, "
                    "forcing cleanup. Thread may be blocked."
                )

        self.ping_thread = None
        self.stop_ping = None
        self.last_ping_tm = float(0)
        self.last_pong_tm = float(0)

    def _send_ping(self) -> None:
        stop_event = self.stop_ping
        if stop_event is None:
            return

        if stop_event.wait(self.ping_interval) or not self.keep_running:
            return

        while self.keep_running and not stop_event.wait(self.ping_interval):
            try:
                sock = self.sock
                if sock:
                    self.last_ping_tm = time.time()
                    sock.ping(self.ping_payload)
            except Exception as exc:
                error(f"ping/pong failed: {exc}")

    def _get_close_args(self, close_frame: Optional[ABNF]) -> List[Any]:
        if close_frame is None:
            return [None, None]

        data = close_frame.data
        if data and len(data) >= 2:
            code = int.from_bytes(data[:2], byteorder="big")
            try:
                reason = data[2:].decode("utf-8")
            except UnicodeDecodeError:
                reason = None
            return [code, reason]

        return [None, None]

    def _callback(self, callback: Optional[Callable], *args: Any) -> None:
        if callback is None:
            return

        try:
            callback(self, *args)
        except Exception as exc:
            error(f"error from callback {callback}: {exc}")
            if self.on_error:
                try:
                    self.on_error(self, exc)
                except Exception as callback_exc:
                    error(f"error from callback {self.on_error}: {callback_exc}")

    def create_dispatcher(
        self,
        ping_timeout: Optional[Union[float, int]],
        dispatcher: Optional[DispatcherBase] = None,
        is_ssl: bool = False,
    ) -> Union[Dispatcher, SSLDispatcher, WrappedDispatcher]:
        if dispatcher:
            if isinstance(dispatcher, DispatcherBase):
                return dispatcher
            return WrappedDispatcher(self, ping_timeout, dispatcher)

        if is_ssl:
            return SSLDispatcher(self, ping_timeout)

        return Dispatcher(self, ping_timeout)

    def run_forever(
        self,
        sockopt: Optional[Tuple] = None,
        sslopt: Optional[dict] = None,
        ping_interval: Union[float, int] = 0,
        ping_timeout: Optional[Union[float, int]] = None,
        ping_payload: str = "",
        http_proxy_host: Optional[str] = None,
        http_proxy_port: Optional[Union[int, str]] = None,
        http_no_proxy: Optional[list] = None,
        http_proxy_auth: Optional[Tuple[str, str]] = None,
        http_proxy_timeout: Optional[float] = None,
        skip_utf8_validation: bool = False,
        host: Optional[str] = None,
        origin: Optional[str] = None,
        dispatcher: Optional[DispatcherBase] = None,
        suppress_origin: bool = False,
        proxy_type: Optional[str] = None,
        reconnect: Optional[int] = None,
    ) -> bool:
        if reconnect is None:
            reconnect = RECONNECT

        if ping_timeout is not None and ping_timeout <= 0:
            raise WebSocketException("Ensure ping_timeout > 0")

        if ping_interval is not None and ping_interval < 0:
            raise WebSocketException("Ensure ping_interval >= 0")

        if (
            ping_timeout is not None
            and ping_interval is not None
            and ping_interval
            and ping_interval <= ping_timeout
        ):
            raise WebSocketException("Ensure ping_interval > ping_timeout")

        if self.sock:
            raise WebSocketException("socket is already opened")

        if sockopt is None:
            sockopt = ()

        if sslopt is None:
            sslopt = {}

        self.keep_running = True
        self.has_errored = False
        self.has_done_teardown = False
        self.last_ping_tm = float(0)
        self.last_pong_tm = float(0)
        self.last_close_frame = None
        self.ping_interval = ping_interval or float(0)
        self.ping_timeout = ping_timeout
        self.ping_payload = ping_payload

        is_ssl = parse_url(self.url)[3]
        dispatcher_obj = self.create_dispatcher(ping_timeout, dispatcher, is_ssl)

        def teardown(close_frame: Optional[ABNF] = None) -> None:
            with self.has_done_teardown_lock:
                if self.has_done_teardown:
                    return
                self.has_done_teardown = True

            self._stop_ping_thread()
            self.keep_running = False

            current_sock = self.sock
            if current_sock is not None:
                try:
                    current_sock.close()
                except Exception:
                    pass

                if close_frame is None:
                    close_frame = current_sock.close_frame
                if close_frame is not None:
                    self.last_close_frame = close_frame

            self.sock = None
            close_status_code, close_reason = self._get_close_args(
                close_frame if close_frame is not None else self.last_close_frame
            )
            self._callback(self.on_close, close_status_code, close_reason)

        def handle_disconnect(exc: BaseException, reconnecting: bool = False) -> None:
            self.has_errored = True
            self._stop_ping_thread()

            if not reconnecting:
                self._callback(self.on_error, exc)

            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                teardown()
                return

            if reconnect:
                if reconnecting:
                    debug(f"{exc} - reconnect")
                else:
                    debug(f"{exc} - reconnect")
                dispatcher_obj.reconnect(reconnect, set_sock)
            else:
                debug(f"{exc} - goodbye")
                teardown()

        def read() -> bool:
            if not self.keep_running:
                teardown()
                return False

            current_sock = self.sock
            if current_sock is None:
                return False

            try:
                opcode, frame = current_sock.recv_data_frame(True)
            except (
                WebSocketConnectionClosedException,
                WebSocketTimeoutException,
                SSLError,
                socket.error,
            ) as exc:
                handle_disconnect(exc, reconnecting=bool(reconnect))
                return False
            except Exception as exc:
                handle_disconnect(exc, reconnecting=bool(reconnect))
                return False

            if opcode == ABNF.OPCODE_CLOSE:
                self.last_close_frame = frame
                teardown(frame)
                return False

            if opcode == ABNF.OPCODE_PING:
                if len(frame.data) < 126:
                    current_sock.pong(frame.data)
                    self._callback(self.on_ping, frame.data)
                    return True
                handle_disconnect(
                    WebSocketException("Ping message is too long"),
                    reconnecting=bool(reconnect),
                )
                return False

            if opcode == ABNF.OPCODE_PONG:
                self.last_pong_tm = time.time()
                self._callback(self.on_pong, frame.data)
                return True

            if opcode == ABNF.OPCODE_CONT and self.on_cont_message:
                self._callback(self.on_data, frame.data, frame.opcode, frame.fin)
                self._callback(self.on_cont_message, frame.data, frame.fin)
                return True

            data: Any = frame.data
            if opcode == ABNF.OPCODE_TEXT and not skip_utf8_validation:
                data = data.decode("utf-8")

            self._callback(self.on_data, data, opcode, True)
            self._callback(self.on_message, data)
            return True

        def check() -> bool:
            if self.ping_timeout is None:
                return True

            if not self.last_ping_tm:
                return True

            now = time.time()
            ping_timed_out = now - self.last_ping_tm > self.ping_timeout
            pong_missing = self.last_pong_tm - self.last_ping_tm < 0
            pong_timed_out = (
                self.last_pong_tm
                and now - self.last_pong_tm > self.ping_timeout
            )

            if ping_timed_out and (pong_missing or pong_timed_out):
                raise WebSocketTimeoutException("ping/pong timed out")

            return True

        def set_sock(reconnecting: bool = False) -> None:
            if reconnecting and self.sock:
                try:
                    self.sock.shutdown()
                except Exception:
                    pass

            self.sock = WebSocket(
                self.get_mask_key,
                sockopt=sockopt,
                sslopt=sslopt,
                fire_cont_frame=self.on_cont_message is not None,
                skip_utf8_validation=skip_utf8_validation,
                enable_multithread=True,
                dispatcher=dispatcher_obj,
            )
            self.sock.settimeout(getdefaulttimeout())

            try:
                self.sock.connect(
                    self.url,
                    header=self.header,
                    cookie=self.cookie,
                    http_proxy_host=http_proxy_host,
                    http_proxy_port=http_proxy_port,
                    http_no_proxy=http_no_proxy,
                    http_proxy_auth=http_proxy_auth,
                    http_proxy_timeout=http_proxy_timeout,
                    subprotocols=self.subprotocols,
                    host=host,
                    origin=origin,
                    suppress_origin=suppress_origin,
                    proxy_type=proxy_type,
                    socket=self.prepared_socket,
                )

                if self.ping_interval:
                    self._start_ping_thread()

                if reconnecting:
                    self._callback(self.on_reconnect)
                else:
                    self._callback(self.on_open)

                dispatcher_obj.read(self.sock.sock, read, check)
            except (
                WebSocketConnectionClosedException,
                WebSocketTimeoutException,
                SSLError,
                socket.error,
            ) as exc:
                handle_disconnect(exc, reconnecting)
            except (KeyboardInterrupt, SystemExit) as exc:
                handle_disconnect(exc, reconnecting)
            except Exception as exc:
                handle_disconnect(exc, reconnecting)

        set_sock()
        return self.has_errored