from __future__ import annotations

import binascii
import json
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .algorithms import (
    Algorithm,
    get_default_algorithms,
    has_crypto,
    requires_cryptography,
)
from .api_jwk import PyJWK
from .exceptions import (
    DecodeError,
    InvalidAlgorithmError,
    InvalidKeyError,
    InvalidSignatureError,
    InvalidTokenError,
)
from .utils import base64url_decode, base64url_encode
from .warnings import InsecureKeyLengthWarning, RemovedInPyjwt3Warning

if TYPE_CHECKING:
    from .algorithms import AllowedPrivateKeys, AllowedPublicKeys
    from .types import SigOptions


_ALGORITHM_UNSET = object()


class PyJWS:
    header_typ = "JWT"

    def __init__(
        self,
        algorithms: Sequence[str] | None = None,
        options: SigOptions | None = None,
    ) -> None:
        self._algorithms = get_default_algorithms()
        self._valid_algs = (
            set(algorithms) if algorithms is not None else set(self._algorithms)
        )

        for name in list(self._algorithms):
            if name not in self._valid_algs:
                del self._algorithms[name]

        self.options: SigOptions = self._get_default_options()
        if options is not None:
            self.options = {**self.options, **options}

    @staticmethod
    def _get_default_options() -> SigOptions:
        return {
            "verify_signature": True,
            "enforce_minimum_key_length": False,
        }

    def register_algorithm(self, alg_id: str, alg_obj: Algorithm) -> None:
        if alg_id in self._algorithms:
            raise ValueError("Algorithm already has a handler.")

        if not isinstance(alg_obj, Algorithm):
            raise TypeError("Object is not of type `Algorithm`")

        self._algorithms[alg_id] = alg_obj
        self._valid_algs.add(alg_id)

    def unregister_algorithm(self, alg_id: str) -> None:
        if alg_id not in self._algorithms:
            raise KeyError(
                "The specified algorithm could not be removed"
                " because it is not registered."
            )

        del self._algorithms[alg_id]
        self._valid_algs.remove(alg_id)

    def get_algorithms(self) -> list[str]:
        return list(self._valid_algs)

    def get_algorithm_by_name(self, alg_name: str) -> Algorithm:
        try:
            return self._algorithms[alg_name]
        except KeyError as exc:
            if not has_crypto and alg_name in requires_cryptography:
                raise NotImplementedError(
                    f"Algorithm '{alg_name}' could not be found. "
                    "Do you have cryptography installed?"
                ) from exc
            raise NotImplementedError("Algorithm not supported") from exc

    def encode(
        self,
        payload: bytes,
        key: AllowedPrivateKeys | PyJWK | str | bytes,
        algorithm: str | None = _ALGORITHM_UNSET,  # type: ignore[assignment]
        headers: dict[str, Any] | None = None,
        json_encoder: type[json.JSONEncoder] | None = None,
        is_payload_detached: bool = False,
        sort_headers: bool = True,
    ) -> str:
        if algorithm is _ALGORITHM_UNSET:
            algorithm_name = key.algorithm_name if isinstance(key, PyJWK) else "HS256"
        elif algorithm is None:
            algorithm_name = key.algorithm_name if isinstance(key, PyJWK) else "none"
        else:
            algorithm_name = algorithm

        if headers:
            if headers.get("alg"):
                algorithm_name = headers["alg"]
            if headers.get("b64") is False:
                is_payload_detached = True

        header: dict[str, Any] = {
            "typ": self.header_typ,
            "alg": algorithm_name,
        }

        if headers:
            self._validate_headers(headers, encoding=True)
            header.update(headers)

        if not header["typ"]:
            del header["typ"]

        if is_payload_detached:
            header["b64"] = False
            critical_headers = header.get("crit", [])
            if not isinstance(critical_headers, list):
                raise InvalidTokenError("Invalid 'crit' header: must be a list")
            if "b64" not in critical_headers:
                header["crit"] = [*critical_headers, "b64"]
        elif "b64" in header:
            del header["b64"]

        encoded_header = base64url_encode(
            json.dumps(
                header,
                separators=(",", ":"),
                cls=json_encoder,
                sort_keys=sort_headers,
            ).encode()
        )

        encoded_payload = payload if is_payload_detached else base64url_encode(payload)
        signing_input = b".".join((encoded_header, encoded_payload))

        algorithm_obj = self.get_algorithm_by_name(algorithm_name)
        if isinstance(key, PyJWK):
            key = key.key
        prepared_key = algorithm_obj.prepare_key(key)

        key_length_message = algorithm_obj.check_key_length(prepared_key)
        if key_length_message:
            if self.options.get("enforce_minimum_key_length", False):
                raise InvalidKeyError(key_length_message)
            warnings.warn(
                key_length_message,
                InsecureKeyLengthWarning,
                stacklevel=2,
            )

        signature = algorithm_obj.sign(signing_input, prepared_key)
        encoded_signature = base64url_encode(signature)

        if is_payload_detached:
            encoded_payload = b""

        return b".".join(
            (encoded_header, encoded_payload, encoded_signature)
        ).decode("utf-8")

    def decode_complete(
        self,
        jwt: str | bytes,
        key: AllowedPublicKeys | PyJWK | str | bytes = "",
        algorithms: Sequence[str] | None = None,
        options: SigOptions | None = None,
        detached_payload: bytes | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs:
            warnings.warn(
                "passing additional kwargs to decode_complete() is deprecated "
                "and will be removed in pyjwt version 3. "
                f"Unsupported kwargs: {tuple(kwargs.keys())}",
                RemovedInPyjwt3Warning,
                stacklevel=2,
            )

        merged_options: SigOptions
        if options is None:
            merged_options = self.options
        else:
            merged_options = {**self.options, **options}

        verify_signature = merged_options["verify_signature"]

        if verify_signature and not algorithms and not isinstance(key, PyJWK):
            raise DecodeError(
                'It is required that you pass in a value for the "algorithms" '
                "argument when calling decode()."
            )

        payload, signing_input, header, signature = self._load(jwt)
        self._validate_headers(header)

        if header.get("b64", True) is False:
            critical_headers = header.get("crit") or []
            if not isinstance(critical_headers, list) or "b64" not in critical_headers:
                raise InvalidTokenError(
                    "The 'b64' header parameter requires 'b64' to be listed in "
                    "'crit'."
                )

            if detached_payload is None:
                raise DecodeError(
                    'It is required that you pass in a value for the '
                    '"detached_payload" argument to decode a message having the '
                    "b64 header set to false."
                )

            payload = detached_payload
            signing_input = b".".join(
                (signing_input.rsplit(b".", 1)[0], detached_payload)
            )

        if verify_signature:
            self._verify_signature(
                signing_input,
                header,
                signature,
                key,
                algorithms,
                merged_options,
            )

        return {
            "payload": payload,
            "header": header,
            "signature": signature,
        }

    def decode(
        self,
        jwt: str | bytes,
        key: AllowedPublicKeys | PyJWK | str | bytes = "",
        algorithms: Sequence[str] | None = None,
        options: SigOptions | None = None,
        detached_payload: bytes | None = None,
        **kwargs: Any,
    ) -> bytes:
        if kwargs:
            warnings.warn(
                "passing additional kwargs to decode() is deprecated and will "
                "be removed in pyjwt version 3. "
                f"Unsupported kwargs: {tuple(kwargs.keys())}",
                RemovedInPyjwt3Warning,
                stacklevel=2,
            )

        return self.decode_complete(
            jwt,
            key,
            algorithms,
            options,
            detached_payload,
            **kwargs,
        )["payload"]

    def _load(
        self, jwt: str | bytes
    ) -> tuple[bytes, bytes, dict[str, Any], bytes]:
        if isinstance(jwt, str):
            jwt = jwt.encode("utf-8")
        elif not isinstance(jwt, bytes):
            raise DecodeError("Invalid token type. Token must be a <class 'bytes'>")

        try:
            signing_input, crypto_segment = jwt.rsplit(b".", 1)
            header_segment, payload_segment = signing_input.split(b".", 1)
        except ValueError as exc:
            raise DecodeError("Not enough segments") from exc

        try:
            header_data = base64url_decode(header_segment)
        except (TypeError, binascii.Error) as exc:
            raise DecodeError("Invalid header padding") from exc

        try:
            header = json.loads(header_data)
        except ValueError as exc:
            raise DecodeError("Invalid header string: must be a json object") from exc

        if not isinstance(header, dict):
            raise DecodeError("Invalid header string: must be a json object")

        try:
            payload = base64url_decode(payload_segment)
        except (TypeError, binascii.Error) as exc:
            raise DecodeError("Invalid payload padding") from exc

        try:
            signature = base64url_decode(crypto_segment)
        except (TypeError, binascii.Error) as exc:
            raise DecodeError("Invalid crypto padding") from exc

        return payload, signing_input, header, signature

    def _verify_signature(
        self,
        signing_input: bytes,
        header: dict[str, Any],
        signature: bytes,
        key: AllowedPublicKeys | PyJWK | str | bytes,
        algorithms: Sequence[str] | None,
        options: SigOptions,
    ) -> None:
        alg = header.get("alg")
        if not alg:
            raise InvalidAlgorithmError("Algorithm not specified")

        if algorithms is None and isinstance(key, PyJWK):
            algorithms = [key.algorithm_name]

        if algorithms is not None and alg not in algorithms:
            raise InvalidAlgorithmError("The specified alg value is not allowed")

        if isinstance(key, PyJWK):
            if alg != key.algorithm_name:
                raise InvalidAlgorithmError("The specified alg value is not allowed")
            key = key.key

        try:
            alg_obj = self.get_algorithm_by_name(alg)
        except NotImplementedError as exc:
            raise InvalidAlgorithmError("Algorithm not supported") from exc

        prepared_key = alg_obj.prepare_key(key)

        key_length_message = alg_obj.check_key_length(prepared_key)
        if key_length_message:
            if options.get("enforce_minimum_key_length", False):
                raise InvalidKeyError(key_length_message)
            warnings.warn(
                key_length_message,
                InsecureKeyLengthWarning,
                stacklevel=2,
            )

        if not alg_obj.verify(signing_input, prepared_key, signature):
            raise InvalidSignatureError("Signature verification failed")

    def _validate_headers(
        self,
        headers: dict[str, Any],
        encoding: bool = False,
    ) -> None:
        if "kid" in headers:
            self._validate_kid(headers["kid"])

        if "crit" in headers and not isinstance(headers["crit"], list):
            raise InvalidTokenError("Invalid 'crit' header: must be a list")

        if encoding and headers.get("b64") is False:
            critical_headers = headers.get("crit", [])
            if not isinstance(critical_headers, list):
                raise InvalidTokenError("Invalid 'crit' header: must be a list")

    def _validate_kid(self, kid: Any) -> None:
        if not isinstance(kid, str):
            raise InvalidTokenError("Key ID header parameter must be a string")


_jws_global_obj = PyJWS()

encode = _jws_global_obj.encode
decode_complete = _jws_global_obj.decode_complete
decode = _jws_global_obj.decode
register_algorithm = _jws_global_obj.register_algorithm
unregister_algorithm = _jws_global_obj.unregister_algorithm
get_algorithm_by_name = _jws_global_obj.get_algorithm_by_name
get_unverified_header = lambda jwt: PyJWS()._load(jwt)[2]