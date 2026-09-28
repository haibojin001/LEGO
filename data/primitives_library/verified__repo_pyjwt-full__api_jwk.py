from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from .algorithms import get_default_algorithms, requires_cryptography
from .exceptions import (
    InvalidKeyError,
    MissingCryptographyError,
    PyJWKError,
    PyJWKSetError,
    PyJWTError,
)
from .types import JWKDict

try:
    from .algorithms import has_crypto
except ImportError:
    try:
        import cryptography
    except ImportError:
        has_crypto = False
    else:
        has_crypto = True


class PyJWK:
    def __init__(self, jwk_data: JWKDict, algorithm: str | None = None) -> None:
        self._jwk_data = jwk_data

        kty = jwk_data.get("kty", None)
        if not kty:
            raise InvalidKeyError(f"kty is not found: {jwk_data}")

        if not algorithm and isinstance(jwk_data, dict):
            algorithm = jwk_data.get("alg", None)

        if not algorithm:
            crv = jwk_data.get("crv", None)

            if kty == "EC":
                if crv == "P-256" or not crv:
                    algorithm = "ES256"
                elif crv == "P-384":
                    algorithm = "ES384"
                elif crv == "P-521":
                    algorithm = "ES512"
                elif crv == "secp256k1":
                    algorithm = "ES256K"
                else:
                    raise InvalidKeyError(f"Unsupported crv: {crv}")
            elif kty == "RSA":
                algorithm = "RS256"
            elif kty == "oct":
                algorithm = "HS256"
            elif kty == "OKP":
                if not crv:
                    raise InvalidKeyError(f"crv is not found: {jwk_data}")
                if crv == "Ed25519":
                    algorithm = "EdDSA"
                else:
                    raise InvalidKeyError(f"Unsupported crv: {crv}")
            else:
                raise InvalidKeyError(f"Unsupported kty: {kty}")

        if algorithm in requires_cryptography and not has_crypto:
            raise MissingCryptographyError(
                f"{algorithm} requires 'cryptography' to be installed."
            )

        self.algorithm_name = algorithm

        try:
            self.Algorithm = get_default_algorithms()[algorithm]
        except KeyError:
            raise PyJWKError(
                f"Unable to find an algorithm for key: {jwk_data}"
            ) from None

        self.key = self.Algorithm.from_jwk(jwk_data)

    @staticmethod
    def from_dict(obj: JWKDict, algorithm: str | None = None) -> PyJWK:
        return PyJWK(obj, algorithm)

    @staticmethod
    def from_json(data: str, algorithm: None = None) -> PyJWK:
        return PyJWK.from_dict(json.loads(data), algorithm)

    @property
    def key_type(self) -> str | None:
        return self._jwk_data.get("kty", None)

    @property
    def key_id(self) -> str | None:
        return self._jwk_data.get("kid", None)

    @property
    def public_key_use(self) -> str | None:
        return self._jwk_data.get("use", None)


class PyJWKSet:
    def __init__(self, keys: list[JWKDict]) -> None:
        self.keys: list[PyJWK] = []

        if not keys:
            raise PyJWKSetError("The JWK Set did not contain any keys")

        if not isinstance(keys, list):
            raise PyJWKSetError("Invalid JWK Set value")

        for key in keys:
            try:
                self.keys.append(PyJWK(key))
            except PyJWTError as error:
                if isinstance(error, MissingCryptographyError):
                    raise error
                continue

        if len(self.keys) == 0:
            raise PyJWKSetError(
                "The JWK Set did not contain any usable keys. Perhaps "
                "'cryptography' is not installed?"
            )

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> PyJWKSet:
        return PyJWKSet(obj.get("keys", []))

    @staticmethod
    def from_json(data: str) -> PyJWKSet:
        return PyJWKSet.from_dict(json.loads(data))

    def __getitem__(self, kid: str) -> PyJWK:
        for key in self.keys:
            if key.key_id == kid:
                return key
        raise KeyError(f"keyset has no key for kid: {kid}")

    def __iter__(self) -> Iterator[PyJWK]:
        return iter(self.keys)


class PyJWTSetWithTimestamp:
    def __init__(self, jwk_set: PyJWKSet):
        self.jwk_set = jwk_set
        self.timestamp = time.monotonic()

    def get_jwk_set(self) -> PyJWKSet:
        return self.jwk_set

    def get_timestamp(self) -> float:
        return self.timestamp