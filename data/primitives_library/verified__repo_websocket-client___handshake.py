import hashlib
import hmac
import os
import socket
from base64 import encodebytes as base64encode
from http import HTTPStatus
from typing import Any, List, Optional

from ._cookiejar import SimpleCookieJar
from ._exceptions import WebSocketBadStatusException, WebSocketException
from ._http import read_headers
from ._logging import dump, error
from ._socket import send

__all__ = ["handshake_response", "handshake", "SUPPORTED_REDIRECT_STATUSES"]

VERSION = 13

SUPPORTED_REDIRECT_STATUSES = (
    HTTPStatus.MOVED_PERMANENTLY,
    HTTPStatus.FOUND,
    HTTPStatus.SEE_OTHER,
    HTTPStatus.TEMPORARY_REDIRECT,
    HTTPStatus.PERMANENT_REDIRECT,
)

SUCCESS_STATUSES = SUPPORTED_REDIRECT_STATUSES + (HTTPStatus.SWITCHING_PROTOCOLS,)

CookieJar = SimpleCookieJar()


class handshake_response:
    def __init__(self, status: int, headers: dict, subprotocol: Optional[str]) -> None:
        self.status = status
        self.headers = headers
        self.subprotocol = subprotocol
        CookieJar.add(headers.get("set-cookie"))


def handshake(
    sock: socket.socket,
    url: str,
    hostname: str,
    port: int,
    resource: str,
    **options: Any,
) -> handshake_response:
    request_headers, key = _get_handshake_headers(
        resource, url, hostname, port, options
    )
    request = "\r\n".join(request_headers)

    send(sock, request)
    dump("request header", request)

    status, response_headers = _get_resp_headers(sock)
    if status in SUPPORTED_REDIRECT_STATUSES:
        return handshake_response(status, response_headers, None)

    valid, subprotocol = _validate(
        response_headers, key, options.get("subprotocols")
    )
    if not valid:
        raise WebSocketException("Invalid WebSocket Header")

    return handshake_response(status, response_headers, subprotocol)


def _pack_hostname(hostname: str) -> str:
    if ":" in hostname:
        return "[" + hostname + "]"
    return hostname


def _get_handshake_headers(
    resource: str, url: str, host: str, port: int, options: dict
) -> tuple:
    result = [
        f"GET {resource} HTTP/1.1",
        "Upgrade: websocket",
    ]

    packed_host = _pack_hostname(host)
    authority = packed_host if port in (80, 443) else f"{packed_host}:{port}"

    if not options.get("suppress_host"):
        chosen_host = options.get("host")
        result.append(f"Host: {chosen_host if chosen_host else authority}")

    scheme, url = url.split(":", 1)
    if not options.get("suppress_origin"):
        if "origin" in options and options["origin"] is not None:
            result.append(f"Origin: {options['origin']}")
        else:
            origin_scheme = "https" if scheme == "wss" else "http"
            result.append(f"Origin: {origin_scheme}://{authority}")

    key = _create_sec_websocket_key()
    supplied_headers = options.get("header")

    if not supplied_headers or "Sec-WebSocket-Key" not in supplied_headers:
        result.append(f"Sec-WebSocket-Key: {key}")
    else:
        key = supplied_headers["Sec-WebSocket-Key"]

    if not supplied_headers or "Sec-WebSocket-Version" not in supplied_headers:
        result.append(f"Sec-WebSocket-Version: {VERSION}")

    connection = options.get("connection")
    result.append(connection if connection else "Connection: Upgrade")

    requested_subprotocols = options.get("subprotocols")
    if requested_subprotocols:
        result.append(
            "Sec-WebSocket-Protocol: " + ",".join(requested_subprotocols)
        )

    if supplied_headers:
        if isinstance(supplied_headers, dict):
            supplied_headers = [
                ": ".join((name, value))
                for name, value in supplied_headers.items()
                if value is not None
            ]
        result.extend(supplied_headers)

    stored_cookie = CookieJar.get(host)
    supplied_cookie = options.get("cookie", None)
    cookies = "; ".join(filter(None, (stored_cookie, supplied_cookie)))
    if cookies:
        result.append(f"Cookie: {cookies}")

    result.extend(("", ""))
    return result, key


def _get_resp_headers(
    sock: socket.socket, success_statuses: tuple = SUCCESS_STATUSES
) -> tuple:
    status, headers, message = read_headers(sock)

    if status in success_statuses:
        return status, headers

    content_length = headers.get("content-length")
    body = None

    if content_length:
        from ._socket import recv

        try:
            remaining = int(content_length)
        except ValueError:
            raise WebSocketException(
                f"Invalid content-length header: {content_length!r}"
            )

        body = b""
        while remaining > 0:
            part = recv(sock, min(remaining, 16384))
            body += part
            remaining -= len(part)

    rendered_body = (
        body.decode("utf-8", errors="replace") if body else None
    )
    detail = (
        f"Handshake status {status} {message} -+-+- {headers} -+-+- "
        f"{rendered_body}"
    )
    raise WebSocketBadStatusException(detail, status, message, headers, body)


_HEADERS_TO_CHECK = {
    "upgrade": "websocket",
    "connection": "upgrade",
}


def _validate(headers: dict, key: str, subprotocols: Optional[List[str]]) -> tuple:
    for field, required_value in _HEADERS_TO_CHECK.items():
        received = headers.get(field, None)
        if not received:
            return False, None

        values = [item.strip().lower() for item in received.split(",")]
        if required_value not in values:
            return False, None

    selected_subprotocol = None
    if subprotocols:
        selected_subprotocol = headers.get("sec-websocket-protocol", None)
        if (
            not selected_subprotocol
            or selected_subprotocol.lower()
            not in [protocol.lower() for protocol in subprotocols]
        ):
            error(f"Invalid subprotocol: {subprotocols}")
            return False, None
        selected_subprotocol = selected_subprotocol.lower()

    accepted = headers.get("sec-websocket-accept", None)
    if not accepted:
        return False, None

    accepted = accepted.lower()
    if isinstance(accepted, str):
        accepted = accepted.encode("utf-8")

    challenge = (
        f"{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    ).encode("utf-8")
    expected = base64encode(hashlib.sha1(challenge).digest()).strip().lower()

    if hmac.compare_digest(expected, accepted):
        return True, selected_subprotocol
    return False, None


def _create_sec_websocket_key() -> str:
    return base64encode(os.urandom(16)).decode("utf-8").strip()