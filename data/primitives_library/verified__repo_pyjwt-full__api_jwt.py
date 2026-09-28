from __future__ import annotations

import json
import os
import warnings
from calendar import timegm
from collections.abc import Container, Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Union, cast

from .api_jws import PyJWS, _ALGORITHM_UNSET, _jws_global_obj
from .exceptions import (
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAudienceError,
    InvalidIssuedAtError,
    InvalidIssuerError,
    InvalidJTIError,
    InvalidSubjectError,
    MissingRequiredClaimError,
)
from .warnings import RemovedInPyjwt3Warning

if TYPE_CHECKING or bool(os.getenv("SPHINX_BUILD", "")):
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from .algorithms import AllowedPrivateKeys, AllowedPublicKeys
    from .api_jwk import PyJWK
    from .types import FullOptions, Options, SigOptions

    AllowedPrivateKeyTypes: TypeAlias = Union[AllowedPrivateKeys, PyJWK, str, bytes]
    AllowedPublicKeyTypes: TypeAlias = Union[AllowedPublicKeys, PyJWK, str, bytes]


class PyJWT:
    def __init__(self, options: Options | None = None) -> None:
        self.options: FullOptions = self._get_default_options()
        if options is not None:
            self.options = self._merge_options(options)
        self._jws = PyJWS(options=self._get_sig_options())

    @staticmethod
    def _get_default_options() -> FullOptions:
        return {
            "verify_signature": True,
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iat": True,
            "verify_aud": True,
            "verify_iss": True,
            "verify_sub": True,
            "verify_jti": True,
            "require": [],
            "strict_aud": False,
            "enforce_minimum_key_length": False,
        }

    def _get_sig_options(self) -> SigOptions:
        return {
            "verify_signature": self.options["verify_signature"],
            "enforce_minimum_key_length": self.options.get(
                "enforce_minimum_key_length", False
            ),
        }

    def _merge_options(self, options: Options | None = None) -> FullOptions:
        if options is None:
            return self.options

        if not options.get("verify_signature", True):
            options["verify_exp"] = options.get("verify_exp", False)
            options["verify_nbf"] = options.get("verify_nbf", False)
            options["verify_iat"] = options.get("verify_iat", False)
            options["verify_aud"] = options.get("verify_aud", False)
            options["verify_iss"] = options.get("verify_iss", False)
            options["verify_sub"] = options.get("verify_sub", False)
            options["verify_jti"] = options.get("verify_jti", False)

        return {**self.options, **options}

    def encode(
        self,
        payload: dict[str, Any],
        key: AllowedPrivateKeyTypes,
        algorithm: str | None = _ALGORITHM_UNSET,  # type: ignore[assignment]
        headers: dict[str, Any] | None = None,
        json_encoder: type[json.JSONEncoder] | None = None,
        sort_headers: bool = True,
    ) -> str:
        if not isinstance(payload, dict):
            raise TypeError(
                "Expecting a dict object, as JWT only supports "
                "JSON objects as payloads."
            )

        payload = payload.copy()
        for claim in ("exp", "iat", "nbf"):
            value = payload.get(claim)
            if isinstance(value, datetime):
                payload[claim] = timegm(value.utctimetuple())

        if "iss" in payload and not isinstance(payload["iss"], str):
            raise TypeError("Issuer (iss) must be a string.")

        encoded_payload = self._encode_payload(
            payload,
            headers=headers,
            json_encoder=json_encoder,
        )

        return self._jws.encode(
            encoded_payload,
            key,
            algorithm,
            headers,
            json_encoder,
            sort_headers=sort_headers,
        )

    def _encode_payload(
        self,
        payload: dict[str, Any],
        headers: dict[str, Any] | None = None,
        json_encoder: type[json.JSONEncoder] | None = None,
    ) -> bytes:
        return json.dumps(
            payload,
            separators=(",", ":"),
            cls=json_encoder,
        ).encode("utf-8")

    def decode_complete(
        self,
        jwt: str | bytes,
        key: AllowedPublicKeyTypes = "",
        algorithms: Sequence[str] | None = None,
        options: Options | None = None,
        verify: bool | None = None,
        detached_payload: bytes | None = None,
        audience: str | Iterable[str] | None = None,
        issuer: str | Container[str] | None = None,
        subject: str | None = None,
        leeway: float | timedelta = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if verify is not None:
            warnings.warn(
                "The `verify` argument to `decode` does nothing in PyJWT 2.0 "
                "and will be removed in PyJWT 3.0. You need to use the "
                "`verify_signature` key in the `options` dictionary instead.",
                RemovedInPyjwt3Warning,
                stacklevel=2,
            )

        if kwargs:
            warnings.warn(
                "passing additional kwargs to decode() is deprecated "
                "and will be removed in pyjwt version 3. Unsupported kwargs: "
                + ", ".join(kwargs),
                RemovedInPyjwt3Warning,
                stacklevel=2,
            )

        options = dict(options or {})

        if not options.get("verify_signature", True):
            options.setdefault("verify_exp", False)
            options.setdefault("verify_nbf", False)
            options.setdefault("verify_iat", False)
            options.setdefault("verify_aud", False)
            options.setdefault("verify_iss", False)
            options.setdefault("verify_sub", False)
            options.setdefault("verify_jti", False)

        decoded = self._jws.decode_complete(
            jwt,
            key,
            algorithms,
            options,
            detached_payload=detached_payload,
        )

        payload = self._decode_payload(decoded)

        merged_options = self._merge_options(options)
        self._validate_claims(
            payload,
            merged_options,
            audience=audience,
            issuer=issuer,
            subject=subject,
            leeway=leeway,
        )

        decoded["payload"] = payload
        return decoded

    def decode(
        self,
        jwt: str | bytes,
        key: AllowedPublicKeyTypes = "",
        algorithms: Sequence[str] | None = None,
        options: Options | None = None,
        verify: bool | None = None,
        detached_payload: bytes | None = None,
        audience: str | Iterable[str] | None = None,
        issuer: str | Container[str] | None = None,
        subject: str | None = None,
        leeway: float | timedelta = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        decoded = self.decode_complete(
            jwt,
            key,
            algorithms,
            options,
            verify,
            detached_payload,
            audience,
            issuer,
            subject,
            leeway,
            **kwargs,
        )
        return cast(dict[str, Any], decoded["payload"])

    def _decode_payload(self, decoded: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = json.loads(decoded["payload"])
        except ValueError as exc:
            raise DecodeError(f"Invalid payload string: {exc}") from exc

        if not isinstance(payload, dict):
            raise DecodeError("Invalid payload string: must be a json object")

        return cast(dict[str, Any], payload)

    def _validate_claims(
        self,
        payload: dict[str, Any],
        options: FullOptions,
        audience: str | Iterable[str] | None = None,
        issuer: str | Container[str] | None = None,
        subject: str | None = None,
        leeway: float | timedelta = 0,
    ) -> None:
        self._validate_required_claims(payload, options)

        if isinstance(leeway, timedelta):
            leeway = leeway.total_seconds()

        now = datetime.now(tz=timezone.utc).timestamp()

        if "iat" in payload and options["verify_iat"]:
            self._validate_iat(payload, now, leeway)

        if "nbf" in payload and options["verify_nbf"]:
            self._validate_nbf(payload, now, leeway)

        if "exp" in payload and options["verify_exp"]:
            self._validate_exp(payload, now, leeway)

        if options["verify_iss"]:
            self._validate_iss(payload, issuer)

        if options["verify_aud"]:
            self._validate_aud(
                payload,
                audience,
                strict=options.get("strict_aud", False),
            )

        if options["verify_sub"]:
            self._validate_sub(payload, subject)

        if options["verify_jti"]:
            self._validate_jti(payload)

    def _validate_required_claims(
        self,
        payload: dict[str, Any],
        options: FullOptions,
    ) -> None:
        for claim in options["require"]:
            if payload.get(claim) is None:
                raise MissingRequiredClaimError(claim)

    def _validate_iat(
        self,
        payload: dict[str, Any],
        now: float,
        leeway: float,
    ) -> None:
        try:
            issued_at = int(payload["iat"])
        except ValueError:
            raise InvalidIssuedAtError(
                "Issued At claim (iat) must be an integer."
            ) from None

        if issued_at > now + leeway:
            raise ImmatureSignatureError("The token is not yet valid (iat)")

    def _validate_nbf(
        self,
        payload: dict[str, Any],
        now: float,
        leeway: float,
    ) -> None:
        try:
            not_before = int(payload["nbf"])
        except ValueError:
            raise DecodeError("Not Before claim (nbf) must be an integer.") from None

        if not_before > now + leeway:
            raise ImmatureSignatureError("The token is not yet valid (nbf)")

    def _validate_exp(
        self,
        payload: dict[str, Any],
        now: float,
        leeway: float,
    ) -> None:
        try:
            expiration = int(payload["exp"])
        except ValueError:
            raise DecodeError("Expiration Time claim (exp) must be an integer.") from None

        if expiration <= now - leeway:
            raise ExpiredSignatureError("Signature has expired")

    def _validate_aud(
        self,
        payload: dict[str, Any],
        audience: str | Iterable[str] | None,
        strict: bool = False,
    ) -> None:
        if audience is None:
            if "aud" not in payload or payload["aud"] in (None, "", []):
                return
            raise InvalidAudienceError("Invalid audience")

        if "aud" not in payload or payload["aud"] in (None, "", []):
            raise MissingRequiredClaimError("aud")

        audience_claims = payload["aud"]

        if strict:
            if not isinstance(audience, str):
                raise InvalidAudienceError("Invalid audience (strict)")

            if not isinstance(audience_claims, str):
                raise InvalidAudienceError("Invalid claim format in token (strict)")

            if audience_claims != audience:
                raise InvalidAudienceError("Audience doesn't match (strict)")

            return

        if isinstance(audience_claims, str):
            audience_claims = [audience_claims]
        elif not isinstance(audience_claims, list):
            raise InvalidAudienceError("Invalid claim format in token")

        if any(not isinstance(item, str) for item in audience_claims):
            raise InvalidAudienceError("Invalid claim format in token")

        if isinstance(audience, str):
            audience = [audience]

        if all(item not in audience for item in audience_claims):
            raise InvalidAudienceError("Audience doesn't match")

    def _validate_iss(
        self,
        payload: dict[str, Any],
        issuer: str | Container[str] | None,
    ) -> None:
        if issuer is None:
            return

        if "iss" not in payload:
            raise MissingRequiredClaimError("iss")

        if isinstance(issuer, str):
            if payload["iss"] != issuer:
                raise InvalidIssuerError("Invalid issuer")
        elif payload["iss"] not in issuer:
            raise InvalidIssuerError("Invalid issuer")

    def _validate_sub(
        self,
        payload: dict[str, Any],
        subject: str | None = None,
    ) -> None:
        if "sub" not in payload:
            return

        if not isinstance(payload["sub"], str):
            raise InvalidSubjectError("Subject must be a string")

        if subject is not None and payload["sub"] != subject:
            raise InvalidSubjectError("Invalid subject")

    def _validate_jti(self, payload: dict[str, Any]) -> None:
        if "jti" not in payload:
            return

        if not isinstance(payload["jti"], str):
            raise InvalidJTIError("JWT ID must be a string")


_jwt_global_obj = PyJWT()
encode = _jwt_global_obj.encode
decode_complete = _jwt_global_obj.decode_complete
decode = _jwt_global_obj.decode