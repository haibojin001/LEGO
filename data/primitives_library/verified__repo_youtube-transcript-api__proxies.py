from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict


class InvalidProxyConfig(Exception):
    pass


class RequestsProxyConfigDict(TypedDict):
    http: str
    https: str


class ProxyConfig(ABC):
    @abstractmethod
    def to_requests_dict(self) -> RequestsProxyConfigDict:
        pass

    @property
    def prevent_keeping_connections_alive(self) -> bool:
        return False

    @property
    def retries_when_blocked(self) -> int:
        return 0


class GenericProxyConfig(ProxyConfig):
    def __init__(
        self,
        http_url: Optional[str] = None,
        https_url: Optional[str] = None,
    ):
        if not http_url and not https_url:
            raise InvalidProxyConfig(
                "GenericProxyConfig requires you to define at least one of the two: "
                "http or https"
            )
        self.http_url = http_url
        self.https_url = https_url

    def to_requests_dict(self) -> RequestsProxyConfigDict:
        return {
            "http": self.http_url or self.https_url,
            "https": self.https_url or self.http_url,
        }


class WebshareProxyConfig(GenericProxyConfig):
    DEFAULT_DOMAIN_NAME = "p.webshare.io"
    DEFAULT_PORT = 80

    def __init__(
        self,
        proxy_username: str,
        proxy_password: str,
        filter_ip_locations: Optional[List[str]] = None,
        retries_when_blocked: int = 10,
        domain_name: str = DEFAULT_DOMAIN_NAME,
        proxy_port: int = DEFAULT_PORT,
    ):
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.domain_name = domain_name
        self.proxy_port = proxy_port
        self._filter_ip_locations = filter_ip_locations or []
        self._retries_when_blocked = retries_when_blocked

    @property
    def url(self) -> str:
        locations = "".join(
            "-" + location.upper() for location in self._filter_ip_locations
        )
        username = self.proxy_username
        rotating_suffix = "-rotate"
        if username.endswith(rotating_suffix):
            username = username[: -len(rotating_suffix)]
        return (
            f"http://{username}{locations}{rotating_suffix}:"
            f"{self.proxy_password}@{self.domain_name}:{self.proxy_port}/"
        )

    @property
    def http_url(self) -> str:
        return self.url

    @property
    def https_url(self) -> str:
        return self.url

    @property
    def prevent_keeping_connections_alive(self) -> bool:
        return True

    @property
    def retries_when_blocked(self) -> int:
        return self._retries_when_blocked