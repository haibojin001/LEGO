import errno
import os
import socket
from base64 import encodebytes as base64encode
from typing import Any, Tuple

from ._exceptions import (
    WebSocketAddressException,
    WebSocketException,
    WebSocketProxyException,
)
from ._logging import debug, dump, trace
from ._socket import DEFAULT_SOCKET_OPTION, recv_line, send
from ._ssl_compat import HAVE_SSL, ssl
from ._url import get_proxy_info, parse_url

__all__ = ["proxy_info", "connect", "read_headers"]

try:
    from python_socks._errors import ProxyConnectionError, ProxyError, ProxyTimeoutError
    from python_socks._types import ProxyType
    from python_socks.sync import Proxy

    HAVE_PYTHON_SOCKS = True
except ImportError:
    HAVE_PYTHON_SOCKS = False

    class ProxyError(Exception):
        pass

    class ProxyTimeoutError(Exception):
        pass

    class ProxyConnectionError(Exception):
        pass

    class ProxyType:
        pass


class proxy_info:
    def __init__(self, **options):
        self.proxy_host = options.get("http_proxy_host", None)
        if self.proxy_host:
            self.proxy_port = options.get("http_proxy_port", 0)
            self.auth = options.get("http_proxy_auth", None)
            self.no_proxy = options.get("http_no_proxy", None)
            self.proxy_protocol = options.get("proxy_type", "http")
            self.proxy_timeout = options.get("http_proxy_timeout", None)
            if self.proxy_protocol not in (
                "http",
                "socks4",
                "socks4a",
                "socks5",
                "socks5h",
            ):
                raise ProxyError(
                    "Only http, socks4, socks5 proxy protocols are supported"
                )
        else:
            self.proxy_port = 0
            self.auth = None
            self.no_proxy = None
            self.proxy_protocol = "http"
            self.proxy_timeout = None


def _start_proxied_socket(
    url: str, options: Any, proxy: Any
) -> Tuple[socket.socket, Tuple[str, int, str]]:
    if proxy.proxy_host and not proxy.proxy_port:
        raise WebSocketProxyException("Cannot use port 0 when proxy_host specified")
    if not HAVE_PYTHON_SOCKS:
        raise WebSocketException(
            "Python Socks is needed for SOCKS proxying but is not available"
        )

    hostname, port, resource, is_secure = parse_url(url)

    if proxy.proxy_protocol == "socks4":
        rdns = False
        proxy_type = ProxyType.SOCKS4
    elif proxy.proxy_protocol == "socks4a":
        rdns = True
        proxy_type = ProxyType.SOCKS4
    elif proxy.proxy_protocol == "socks5":
        rdns = False
        proxy_type = ProxyType.SOCKS5
    elif proxy.proxy_protocol == "socks5h":
        rdns = True
        proxy_type = ProxyType.SOCKS5

    ws_proxy = Proxy.create(
        proxy_type=proxy_type,
        host=proxy.proxy_host,
        port=int(proxy.proxy_port),
        username=proxy.auth[0] if proxy.auth else None,
        password=proxy.auth[1] if proxy.auth else None,
        rdns=rdns,
    )
    sock = ws_proxy.connect(hostname, port, timeout=proxy.proxy_timeout)

    if is_secure:
        if not HAVE_SSL:
            raise WebSocketException("SSL not available.")
        sock = _ssl_socket(sock, options.sslopt, hostname)

    return sock, (hostname, port, resource)


def connect(
    url: str, options: Any, proxy: Any, socket: Any
) -> Tuple[socket.socket, Tuple[str, int, str]]:
    if proxy.proxy_host and not socket and proxy.proxy_protocol != "http":
        return _start_proxied_socket(url, options, proxy)

    hostname, port_from_url, resource, is_secure = parse_url(url)

    if socket:
        return socket, (hostname, port_from_url, resource)

    addrinfo_list, need_tunnel, auth = _get_addrinfo_list(
        hostname, port_from_url, is_secure, proxy
    )
    if not addrinfo_list:
        raise WebSocketException(f"Host not found.: {hostname}:{port_from_url}")

    sock = None
    try:
        sock = _open_socket(addrinfo_list, options.sockopt, options.timeout)
        if need_tunnel:
            sock = _tunnel(sock, hostname, port_from_url, auth)

        if is_secure:
            if not HAVE_SSL:
                raise WebSocketException("SSL not available.")
            sock = _ssl_socket(sock, options.sslopt, hostname)

        return sock, (hostname, port_from_url, resource)
    except:
        if sock:
            sock.close()
        raise


def _get_addrinfo_list(
    hostname: str, port: int, is_secure: bool, proxy: Any
) -> Tuple[list, bool, Any]:
    try:
        phost, pport, pauth = get_proxy_info(
            hostname,
            is_secure,
            proxy.proxy_host,
            proxy.proxy_port,
            proxy.auth,
            proxy.no_proxy,
        )
    except TypeError as e:
        raise WebSocketAddressException(e)

    try:
        if not phost:
            addresses = socket.getaddrinfo(
                hostname, port, 0, socket.SOCK_STREAM, socket.SOL_TCP
            )
            return addresses, False, None

        pport = pport or 80
        addresses = socket.getaddrinfo(
            phost, pport, 0, socket.SOCK_STREAM, socket.SOL_TCP
        )
        return addresses, True, pauth
    except (socket.gaierror, TypeError) as e:
        raise WebSocketAddressException(e)


def _open_socket(addrinfo_list, sockopt, timeout):
    err = None
    for addrinfo in addrinfo_list:
        family, socktype, proto = addrinfo[:3]
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)

        for option in DEFAULT_SOCKET_OPTION:
            sock.setsockopt(*option)
        for option in sockopt:
            sock.setsockopt(*option)

        address = addrinfo[4]
        err = None
        while not err:
            try:
                sock.connect(address)
            except socket.error as error:
                sock.close()
                error.remote_ip = str(address[0])
                refused_errors = (
                    errno.ECONNREFUSED,
                    getattr(errno, "WSAECONNREFUSED", errno.ECONNREFUSED),
                    errno.ENETUNREACH,
                )
                if error.errno not in refused_errors:
                    raise error
                err = error
                continue
            else:
                break
        else:
            continue
        break
    else:
        if err:
            raise err

    return sock


def _wrap_sni_socket(
    sock: socket.socket, sslopt: dict, hostname: str, check_hostname: bool
) -> Any:
    context = sslopt.get("context", None)
    if not context:
        context = ssl.SSLContext(sslopt.get("ssl_version", ssl.PROTOCOL_TLS_CLIENT))

        keylog_file = os.environ.get("SSLKEYLOGFILE")
        if keylog_file is not None:
            context.keylog_filename = keylog_file

        cert_reqs = sslopt.get("cert_reqs", ssl.CERT_REQUIRED)
        if cert_reqs == ssl.CERT_NONE:
            context.check_hostname = False
            context.verify_mode = cert_reqs
        else:
            context.verify_mode = cert_reqs
            context.check_hostname = check_hostname

        if sslopt.get("ca_certs"):
            context.load_verify_locations(cafile=sslopt["ca_certs"])
        if sslopt.get("ca_cert_path"):
            context.load_verify_locations(capath=sslopt["ca_cert_path"])
        if sslopt.get("certfile"):
            context.load_cert_chain(
                sslopt["certfile"],
                sslopt.get("keyfile"),
                sslopt.get("password"),
            )
        if sslopt.get("ciphers"):
            context.set_ciphers(sslopt["ciphers"])

    return context.wrap_socket(
        sock,
        server_hostname=hostname,
        suppress_ragged_eofs=sslopt.get("suppress_ragged_eofs", True),
    )


def _wrap_socket(sock: socket.socket, sslopt: dict, hostname: str) -> Any:
    options = {
        "keyfile": sslopt.get("keyfile", None),
        "certfile": sslopt.get("certfile", None),
        "server_side": sslopt.get("server_side", False),
        "cert_reqs": sslopt.get("cert_reqs", ssl.CERT_REQUIRED),
        "ssl_version": sslopt.get("ssl_version", ssl.PROTOCOL_TLS_CLIENT),
        "ca_certs": sslopt.get("ca_certs", None),
        "do_handshake_on_connect": sslopt.get("do_handshake_on_connect", True),
        "suppress_ragged_eofs": sslopt.get("suppress_ragged_eofs", True),
        "ciphers": sslopt.get("ciphers", None),
    }
    return ssl.wrap_socket(sock, **options)


def _ssl_socket(sock: socket.socket, sslopt: dict, hostname: str) -> Any:
    if sslopt.get("server_hostname", None):
        hostname = sslopt["server_hostname"]

    check_hostname = sslopt.get("check_hostname", True)
    if check_hostname and sslopt.get("cert_reqs", ssl.CERT_REQUIRED) == ssl.CERT_NONE:
        check_hostname = False

    if getattr(ssl, "HAS_SNI", True):
        return _wrap_sni_socket(sock, sslopt, hostname, check_hostname)
    return _wrap_socket(sock, sslopt, hostname)


def _tunnel(sock: socket.socket, host: str, port: int, auth: Any) -> socket.socket:
    connect = [f"CONNECT {host}:{port} HTTP/1.1", f"Host: {host}:{port}"]

    if auth:
        credentials = base64encode(f"{auth[0]}:{auth[1]}".encode("utf-8"))
        credentials = credentials.decode("utf-8").replace("\n", "")
        connect.append(f"Proxy-Authorization: Basic {credentials}")

    connect.append("")
    connect.append("")
    dump("request header", connect)
    send(sock, "\r\n".join(connect))

    status, _, _ = read_headers(sock)
    if status != 200:
        raise WebSocketProxyException(f"failed CONNECT via proxy status: {status}")

    return sock


def read_headers(sock: socket.socket) -> Tuple[int, dict, str]:
    status_line = recv_line(sock).decode("utf-8").strip()
    if not status_line:
        raise WebSocketException("Handshake status 0")

    status_info = status_line.split(" ", 2)
    if len(status_info) != 3:
        raise WebSocketException("Invalid status line")

    try:
        status = int(status_info[1])
    except ValueError:
        raise WebSocketException("Invalid status code")

    headers = {}
    trace("--- response header ---")
    trace(status_line)

    while True:
        line = recv_line(sock).decode("utf-8").strip()
        if not line:
            break

        trace(line)
        key_value = line.split(":", 1)
        if len(key_value) != 2:
            raise WebSocketException("Invalid header")

        key, value = key_value
        headers[key.lower()] = value.strip()

    trace("-----------------------")
    return status, headers, status_info[2]