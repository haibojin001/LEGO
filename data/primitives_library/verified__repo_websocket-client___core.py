import socket
import struct
import threading
import time
from typing import Any, Callable, Optional, Type, Union
from urllib.parse import urljoin

from ._abnf import ABNF, STATUS_NORMAL, continuous_frame, frame_buffer
from ._exceptions import (
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketProtocolException,
    WebSocketTimeoutException,
)
from ._handshake import SUPPORTED_REDIRECT_STATUSES, handshake, handshake_response
from ._http import connect, proxy_info
from ._logging import debug, error, isEnabledForError, isEnabledForTrace, trace
from ._socket import getdefaulttimeout, recv, send, sock_opt
from ._ssl_compat import ssl
from ._utils import NoLock
from ._dispatcher import DispatcherBase, WrappedDispatcher

__all__ = ["WebSocket", "create_connection"]


def _normalize_close_reason(reason: Union[str, bytes, None]) -> bytes:
    if reason is None:
        return b""
    if isinstance(reason, str):
        return reason.encode("utf-8")
    if isinstance(reason, bytes):
        return reason
    return bytes(reason)


class WebSocket:
    def __init__(
        self,
        get_mask_key: Optional[Callable] = None,
        sockopt: Optional[list] = None,
        sslopt: Optional[dict] = None,
        fire_cont_frame: bool = False,
        enable_multithread: bool = True,
        skip_utf8_validation: bool = False,
        dispatcher: Optional[Union[DispatcherBase, WrappedDispatcher]] = None,
        **_: Any,
    ) -> None:
        self.sock_opt = sock_opt(sockopt, sslopt)
        self.handshake_response: Optional[handshake_response] = None
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self.close_frame: Optional[ABNF] = None
        self.get_mask_key = get_mask_key
        self.frame_buffer = frame_buffer(self._recv, skip_utf8_validation)
        self.cont_frame = continuous_frame(fire_cont_frame, skip_utf8_validation)
        self.dispatcher = dispatcher

        if enable_multithread:
            self.lock = threading.Lock()
            self.readlock = threading.Lock()
        else:
            self.lock = NoLock()
            self.readlock = NoLock()

    def __iter__(self):
        while True:
            yield self.recv()

    def __next__(self):
        return self.recv()

    def next(self):
        return self.__next__()

    def fileno(self):
        if self.sock is None:
            raise WebSocketException("Connection not established")
        return self.sock.fileno()

    def set_mask_key(self, func):
        self.get_mask_key = func

    def gettimeout(self) -> Optional[Union[float, int]]:
        return self.sock_opt.timeout

    def settimeout(self, timeout: Optional[Union[float, int]]) -> None:
        self.sock_opt.timeout = timeout
        if self.sock is not None:
            self.sock.settimeout(timeout)

    timeout = property(gettimeout, settimeout)

    def getsubprotocol(self) -> Optional[str]:
        if self.handshake_response is None:
            return None
        return self.handshake_response.subprotocol

    subprotocol = property(getsubprotocol)

    def getstatus(self) -> Optional[int]:
        if self.handshake_response is None:
            return None
        return self.handshake_response.status

    status = property(getstatus)

    def getheaders(self) -> Optional[dict]:
        if self.handshake_response is None:
            return None
        return self.handshake_response.headers

    headers = property(getheaders)

    def is_ssl(self):
        try:
            return isinstance(self.sock, ssl.SSLSocket)
        except (AttributeError, NameError):
            return False

    def connect(self, url, **options):
        self.sock_opt.timeout = options.get("timeout", self.sock_opt.timeout)
        redirect_limit = options.get("redirect_limit", 3)
        supplied_socket = options.pop("socket", None)

        try:
            while True:
                self.sock, addrs = connect(
                    url,
                    self.sock_opt,
                    proxy_info(**options),
                    supplied_socket,
                )
                supplied_socket = None

                self.handshake_response = handshake(self.sock, url, *addrs, **options)
                self.connected = True

                if self.handshake_response.status not in SUPPORTED_REDIRECT_STATUSES:
                    break

                if redirect_limit <= 0:
                    break

                location = self.handshake_response.headers["location"]
                url = urljoin(url, location)
                redirect_limit -= 1

                if self.sock is not None:
                    self.sock.close()
                self.sock = None
                self.connected = False

            return self
        except Exception:
            self.connected = False
            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = None
            raise

    def send(self, payload: Union[bytes, str], opcode: int = ABNF.OPCODE_TEXT) -> int:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return self.send_frame(ABNF.create_frame(payload, opcode))

    def send_frame(self, frame: ABNF) -> int:
        if self.get_mask_key is not None:
            frame.get_mask_key = self.get_mask_key

        data = frame.format()
        length = len(data)

        if isEnabledForTrace():
            trace(f"++Sent raw: {repr(data)}")
            trace(f"++Sent decoded: {frame}")

        with self.lock:
            while data:
                sent = self._send(data)
                data = data[sent:]

        return length

    def send_binary(self, payload: bytes) -> int:
        return self.send(payload, ABNF.OPCODE_BINARY)

    def send_text(self, payload: Union[str, bytes]) -> int:
        return self.send(payload, ABNF.OPCODE_TEXT)

    def send_bytes(self, payload: bytes) -> int:
        return self.send(payload, ABNF.OPCODE_BINARY)

    def ping(self, payload: Union[str, bytes] = "") -> None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PING)

    def pong(self, payload: Union[str, bytes] = "") -> None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PONG)

    def recv(self) -> Union[str, bytes]:
        with self.readlock:
            opcode, data = self.recv_data()

        if opcode == ABNF.OPCODE_TEXT:
            return data.decode("utf-8")
        if opcode == ABNF.OPCODE_BINARY:
            return data
        return ""

    def recv_data(self, control_frame: bool = False):
        opcode, frame = self.recv_data_frame(control_frame)
        if frame is None:
            return opcode, None
        return opcode, frame.data

    def recv_data_frame(self, control_frame: bool = False):
        while True:
            frame = self.recv_frame()

            if not frame:
                return ABNF.OPCODE_CLOSE, None

            if isEnabledForTrace():
                trace(f"++Rcv raw: {repr(frame.data)}")
                trace(f"++Rcv decoded: {frame}")

            if frame.opcode in (
                ABNF.OPCODE_TEXT,
                ABNF.OPCODE_BINARY,
                ABNF.OPCODE_CONT,
            ):
                self.cont_frame.validate(frame)

                if not frame.fin or frame.opcode == ABNF.OPCODE_CONT:
                    frame = self.cont_frame.extract(frame)

                if frame is not None:
                    return frame.opcode, frame

            elif frame.opcode == ABNF.OPCODE_CLOSE:
                self.connected = False
                self.close_frame = frame
                return frame.opcode, frame

            elif frame.opcode == ABNF.OPCODE_PING:
                if len(frame.data) >= 126:
                    raise WebSocketProtocolException("Ping message is too long")

                self.pong(frame.data)
                if control_frame:
                    return frame.opcode, frame

            elif frame.opcode == ABNF.OPCODE_PONG:
                if control_frame:
                    return frame.opcode, frame

    def recv_frame(self):
        return self.frame_buffer.recv_frame()

    def _recv(self, bufsize):
        if not self.connected or self.sock is None:
            raise WebSocketConnectionClosedException("socket is already closed.")
        return recv(self.sock, bufsize)

    def _send(self, payload):
        if not self.connected or self.sock is None:
            raise WebSocketConnectionClosedException("socket is already closed.")
        return send(self.sock, payload)

    def close(
        self,
        status: int = STATUS_NORMAL,
        reason: Union[str, bytes, None] = b"",
        timeout: Optional[Union[int, float]] = 3,
    ) -> None:
        if not self.connected:
            return

        status_max = getattr(ABNF, "STATUS_CODE_MAX", 5000)
        if status < 0 or status >= status_max:
            raise ValueError("status is invalid")

        payload = struct.pack("!H", status) + _normalize_close_reason(reason)

        try:
            self.connected = False
            self.send(payload, ABNF.OPCODE_CLOSE)

            sock = self.sock
            if sock is not None:
                sock.settimeout(timeout)
                started = time.time()

                while sock is not None and (
                    timeout is None or time.time() - started < timeout
                ):
                    try:
                        frame = self.recv_frame()

                        if frame.opcode == ABNF.OPCODE_CLOSE:
                            sock.shutdown(socket.SHUT_RDWR)
                            return

                        if frame.opcode == ABNF.OPCODE_PING:
                            self.pong(frame.data)
                    except (
                        WebSocketTimeoutException,
                        WebSocketConnectionClosedException,
                    ):
                        break
                    except OSError:
                        break
        except Exception:
            if isEnabledForError():
                error("error from closing socket", exc_info=True)
        finally:
            self.shutdown()

    def abort(self) -> None:
        if not self.connected:
            return

        self.connected = False
        if self.sock is not None:
            self.sock.shutdown(socket.SHUT_RDWR)

    def shutdown(self) -> None:
        sock = self.sock
        self.sock = None
        self.connected = False

        if sock is None:
            return

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, socket.error):
            pass

        try:
            sock.close()
        except (OSError, socket.error):
            pass


def create_connection(
    url: str,
    timeout: Optional[Union[int, float]] = None,
    class_: Type[WebSocket] = WebSocket,
    **options,
) -> WebSocket:
    if timeout is None:
        timeout = getdefaulttimeout()

    websock = class_(timeout=timeout, **options)
    websock.connect(url, **options)
    return websock