from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from ssl import SSLContext
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from .api_jwk import PyJWK, PyJWKSet
from .api_jwt import decode_complete as decode_token
from .exceptions import PyJWKClientConnectionError, PyJWKClientError
from .jwk_set_cache import JWKSetCache


class PyJWKClient:
    def __init__(
        self,
        uri: str,
        cache_keys: bool = False,
        max_cached_keys: int = 16,
        cache_jwk_set: bool = True,
        lifespan: float = 300,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,
        ssl_context: SSLContext | None = None,
    ):
        scheme = urlparse(uri).scheme.lower()
        if scheme not in ("http", "https"):
            raise PyJWKClientError(
                f"Invalid JWKS URI scheme {scheme!r}: only 'http' and 'https' "
                f"are supported."
            )

        self.uri = uri
        self.headers = {} if headers is None else headers
        self.timeout = timeout
        self.ssl_context = ssl_context
        self.jwk_set_cache: JWKSetCache | None = None

        if cache_jwk_set:
            if lifespan <= 0:
                raise PyJWKClientError(
                    f'Lifespan must be greater than 0, the input is "{lifespan}"'
                )
            self.jwk_set_cache = JWKSetCache(lifespan)

        if cache_keys:
            self.get_signing_key = lru_cache(maxsize=max_cached_keys)(
                self.get_signing_key
            )

    def fetch_data(self) -> Any:
        try:
            request = urllib.request.Request(self.uri, headers=self.headers)
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
                context=self.ssl_context,
            ) as response:
                data = json.load(response)
        except (URLError, TimeoutError) as error:
            if isinstance(error, HTTPError):
                error.close()
            raise PyJWKClientConnectionError(
                f'Fail to fetch data from the url, err: "{error}"'
            ) from error

        if self.jwk_set_cache is not None:
            self.jwk_set_cache.put(data)

        return data

    def get_jwk_set(self, refresh: bool = False) -> PyJWKSet:
        data = None

        if not refresh and self.jwk_set_cache is not None:
            data = self.jwk_set_cache.get()

        if data is None:
            data = self.fetch_data()

        if not isinstance(data, dict):
            raise PyJWKClientError("The JWKS endpoint did not return a JSON object")

        return PyJWKSet.from_dict(data)

    def get_signing_keys(self, refresh: bool = False) -> list[PyJWK]:
        jwk_set = self.get_jwk_set(refresh)
        keys = [
            key
            for key in jwk_set.keys
            if key.public_key_use in ("sig", None) and key.key_id
        ]

        if not keys:
            raise PyJWKClientError("The JWKS endpoint did not contain any signing keys")

        return keys

    def get_signing_key(self, kid: str) -> PyJWK:
        keys = self.get_signing_keys()
        key = self.match_kid(keys, kid)

        if key is None:
            keys = self.get_signing_keys(refresh=True)
            key = self.match_kid(keys, kid)

            if key is None:
                raise PyJWKClientError(
                    f'Unable to find a signing key that matches: "{kid}"'
                )

        return key

    def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        decoded = decode_token(token, options={"verify_signature": False})
        return self.get_signing_key(decoded["header"].get("kid"))

    @staticmethod
    def match_kid(signing_keys: list[PyJWK], kid: str) -> PyJWK | None:
        for signing_key in signing_keys:
            if signing_key.key_id == kid:
                return signing_key
        return None