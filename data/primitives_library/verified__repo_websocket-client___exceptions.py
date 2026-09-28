from typing import Optional


class WebSocketException(Exception):
    """
    Base exception for WebSocket-related errors.
    """


class WebSocketProtocolException(WebSocketException):
    """
    Raised when a WebSocket protocol violation is detected.
    """


class WebSocketPayloadException(WebSocketException):
    """
    Raised when a WebSocket payload is invalid.
    """


class WebSocketConnectionClosedException(WebSocketException):
    """
    Raised when the connection closes unexpectedly or a network error occurs.
    """


class WebSocketTimeoutException(WebSocketException):
    """
    Raised when a socket operation times out.
    """


class WebSocketProxyException(WebSocketException):
    """
    Raised when a proxy-related error occurs.
    """


class WebSocketBadStatusException(WebSocketException):
    """
    Raised when the server returns an unacceptable handshake status.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        status_message: Optional[str] = None,
        resp_headers: Optional[dict] = None,
        resp_body: Optional[bytes] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.status_message = status_message
        self.resp_headers = resp_headers
        self.resp_body = resp_body


class WebSocketAddressException(WebSocketException):
    """
    Raised when WebSocket address information cannot be resolved.
    """