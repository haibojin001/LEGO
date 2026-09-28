import ipaddress
import os
from typing import Optional, Union
from urllib.parse import unquote, urlparse

from ._exceptions import WebSocketProxyException

__all__ = ["parse_url", "get_proxy_info"]


def parse_url(url: str) -> tuple:
    if ":" not in url:
        raise ValueError("url is invalid")

    scheme, remainder = url.split(":", 1)
    result = urlparse(remainder, scheme="http")

    hostname = result.hostname
    if not hostname:
        raise ValueError("hostname is invalid")

    port = result.port or 0
    secure = False

    if scheme == "ws":
        if not port:
            port = 80
    elif scheme == "wss":
        secure = True
        if not port:
            port = 443
    else:
        raise ValueError("scheme %s is invalid" % scheme)

    resource = result.path or "/"
    if result.query:
        resource += "?" + result.query

    return hostname, port, resource, secure


def _is_ip_address(addr: str) -> bool:
    if not isinstance(addr, str):
        raise TypeError("_is_ip_address() argument 1 must be str")

    try:
        ipaddress.ip_address(addr)
    except ValueError:
        return False
    return True


def _is_subnet_address(hostname: str) -> bool:
    try:
        ipaddress.ip_network(hostname)
    except ValueError:
        return False
    return True


def _is_address_in_network(ip: str, net: str) -> bool:
    try:
        source: Union[ipaddress.IPv4Network, ipaddress.IPv6Network]
        destination: Union[ipaddress.IPv4Network, ipaddress.IPv6Network]
        source = ipaddress.ip_network(ip)
        destination = ipaddress.ip_network(net)
        return source.subnet_of(destination)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _is_no_proxy_host(hostname: str, no_proxy: Optional[list[str]]) -> bool:
    if not no_proxy:
        configured = os.environ.get(
            "no_proxy", os.environ.get("NO_PROXY", "")
        ).replace(" ", "")
        if configured:
            no_proxy = configured.split(",")

    if not no_proxy:
        no_proxy = []

    if "*" in no_proxy or hostname in no_proxy:
        return True

    if _is_ip_address(hostname):
        return any(
            _is_address_in_network(hostname, entry)
            for entry in no_proxy
            if _is_subnet_address(entry)
        )

    for entry in no_proxy:
        if entry.startswith("."):
            if hostname.endswith(entry.lstrip(".")):
                return True
        elif hostname == entry or hostname.endswith("." + entry):
            return True

    return False


def get_proxy_info(
    hostname: str,
    is_secure: bool,
    proxy_host: Optional[str] = None,
    proxy_port: int = 0,
    proxy_auth: Optional[tuple] = None,
    no_proxy: Optional[list[str]] = None,
    proxy_type: str = "http",
) -> tuple:
    if _is_no_proxy_host(hostname, no_proxy):
        return None, 0, None

    if proxy_host:
        if not proxy_port:
            raise WebSocketProxyException(
                "Cannot use port 0 when proxy_host specified"
            )
        return proxy_host, proxy_port, proxy_auth

    variable = "https_proxy" if is_secure else "http_proxy"
    setting = os.environ.get(
        variable, os.environ.get(variable.upper(), "")
    ).replace(" ", "")

    if setting:
        parsed = urlparse(setting)
        credentials = None
        if parsed.username:
            credentials = (
                unquote(parsed.username or ""),
                unquote(parsed.password or ""),
            )
        return parsed.hostname, parsed.port, credentials

    return None, 0, None