from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Union, cast, get_args, overload

from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
    base64url_decode,
    base64url_encode,
    der_to_raw_signature,
    force_bytes,
    from_base64url_uint,
    is_pem_format,
    is_ssh_key,
    raw_to_der_signature,
    to_base64url_uint,
)

try:
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.ec import (
        ECDSA,
        SECP256K1,
        SECP256R1,
        SECP384R1,
        SECP521R1,
        EllipticCurve,
        EllipticCurvePrivateKey,
        EllipticCurvePrivateNumbers,
        EllipticCurvePublicKey,
        EllipticCurvePublicNumbers,
    )
    from cryptography.hazmat.primitives.asymmetric.ed448 import (
        Ed448PrivateKey,
        Ed448PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.rsa import (
        RSAPrivateKey,
        RSAPrivateNumbers,
        RSAPublicKey,
        RSAPublicNumbers,
        rsa_crt_dmp1,
        rsa_crt_dmq1,
        rsa_crt_iqmp,
        rsa_recover_prime_factors,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
        load_pem_private_key,
        load_pem_public_key,
        load_ssh_public_key,
    )

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    AllowedRSAKeys: TypeAlias = Union[RSAPrivateKey, RSAPublicKey]
    AllowedECKeys: TypeAlias = Union[EllipticCurvePrivateKey, EllipticCurvePublicKey]
    AllowedOKPKeys: TypeAlias = Union[
        Ed25519PrivateKey, Ed25519PublicKey, Ed448PrivateKey, Ed448PublicKey
    ]
    AllowedKeys: TypeAlias = Union[AllowedRSAKeys, AllowedECKeys, AllowedOKPKeys]
    AllowedPrivateKeys: TypeAlias = Union[
        RSAPrivateKey, EllipticCurvePrivateKey, Ed25519PrivateKey, Ed448PrivateKey
    ]
    AllowedPublicKeys: TypeAlias = Union[
        RSAPublicKey, EllipticCurvePublicKey, Ed25519PublicKey, Ed448PublicKey
    ]

    if TYPE_CHECKING or bool(os.getenv("SPHINX_BUILD", "")):
        from cryptography.hazmat.primitives.asymmetric.types import (
            PrivateKeyTypes,
            PublicKeyTypes,
        )

    has_crypto = True
except ModuleNotFoundError:
    if sys.version_info >= (3, 11):
        from typing import Never
    else:
        from typing_extensions import Never

    AllowedRSAKeys = Never  # type: ignore[misc]
    AllowedECKeys = Never  # type: ignore[misc]
    AllowedOKPKeys = Never  # type: ignore[misc]
    AllowedKeys = Never  # type: ignore[misc]
    AllowedPrivateKeys = Never  # type: ignore[misc]
    AllowedPublicKeys = Never  # type: ignore[misc]
    has_crypto = False


requires_cryptography = {
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES256K",
    "ES384",
    "ES521",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
    "EdDSA",
}


class Algorithm(ABC):
    _crypto_key_types: tuple[type[AllowedKeys], ...] | None = None

    def compute_hash_digest(self, bytestr: bytes) -> bytes:
        hash_alg = getattr(self, "hash_alg", None)
        if hash_alg is None:
            raise NotImplementedError
        if (
            has_crypto
            and isinstance(hash_alg, type)
            and issubclass(hash_alg, hashes.HashAlgorithm)
        ):
            value = hashes.Hash(hash_alg(), backend=default_backend())
            value.update(bytestr)
            return bytes(value.finalize())
        return bytes(hash_alg(bytestr).digest())

    def check_crypto_key_type(self, key: Any) -> None:
        if not has_crypto or self._crypto_key_types is None:
            raise ValueError(
                "This method requires the cryptography library, and should only be used by cryptography-based algorithms."
            )
        if not isinstance(key, self._crypto_key_types):
            names = (item.__name__ for item in self._crypto_key_types)
            raise InvalidKeyError(
                f"Expected one of {names}, got: {key.__class__.__name__}. Invalid Key type for {self.__class__.__name__}"
            )

    @abstractmethod
    def prepare_key(self, key: Any) -> Any:
        pass

    @abstractmethod
    def sign(self, msg: bytes, key: Any) -> bytes:
        pass

    @abstractmethod
    def verify(self, msg: bytes, key: Any, sig: bytes) -> bool:
        pass

    @overload
    @staticmethod
    @abstractmethod
    def to_jwk(key_obj: Any, as_dict: Literal[True]) -> JWKDict:
        pass

    @overload
    @staticmethod
    @abstractmethod
    def to_jwk(key_obj: Any, as_dict: Literal[False] = False) -> str:
        pass

    @staticmethod
    @abstractmethod
    def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
        pass

    @staticmethod
    @abstractmethod
    def from_jwk(jwk: str | JWKDict) -> Any:
        pass

    def check_key_length(self, key: Any) -> str | None:
        return None


class NoneAlgorithm(Algorithm):
    def prepare_key(self, key: Any) -> None:
        if key == "":
            key = None
        if key is not None:
            raise InvalidKeyError('When alg = "none", key value must be None.')
        return None

    def sign(self, msg: bytes, key: None) -> bytes:
        return b""

    def verify(self, msg: bytes, key: Any, sig: bytes) -> bool:
        return False

    @staticmethod
    def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
        raise NotImplementedError

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> Any:
        raise NotImplementedError


class HMACAlgorithm(Algorithm):
    SHA256: ClassVar[HashlibHash] = hashlib.sha256
    SHA384: ClassVar[HashlibHash] = hashlib.sha384
    SHA512: ClassVar[HashlibHash] = hashlib.sha512

    def __init__(self, hash_alg: HashlibHash) -> None:
        self.hash_alg = hash_alg

    def prepare_key(self, key: Any) -> bytes:
        key = force_bytes(key)
        if is_pem_format(key) or is_ssh_key(key):
            raise InvalidKeyError(
                "The specified key is an asymmetric key or x509 certificate and should not be used as an HMAC secret."
            )
        return key

    def sign(self, msg: bytes, key: bytes) -> bytes:
        return hmac.new(key, msg, self.hash_alg).digest()

    def verify(self, msg: bytes, key: bytes, sig: bytes) -> bool:
        return hmac.compare_digest(sig, self.sign(msg, key))

    @staticmethod
    def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
        key = force_bytes(key_obj)
        result: JWKDict = {
            "kty": "oct",
            "k": base64url_encode(key).decode(),
        }
        return result if as_dict else json.dumps(result)

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> bytes:
        if isinstance(jwk, str):
            jwk = json.loads(jwk)
        if jwk.get("kty") != "oct":
            raise InvalidKeyError("Incorrect key type. Expected: 'oct', Received: %s" % jwk.get("kty"))
        if "k" not in jwk:
            raise InvalidKeyError("Key is missing 'k' parameter")
        return base64url_decode(jwk["k"])


if has_crypto:

    class RSAAlgorithm(Algorithm):
        SHA256: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA256
        SHA384: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA384
        SHA512: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA512
        _crypto_key_types = (RSAPrivateKey, RSAPublicKey)

        def __init__(self, hash_alg: type[hashes.HashAlgorithm]) -> None:
            self.hash_alg = hash_alg

        def prepare_key(self, key: Any) -> AllowedRSAKeys:
            if isinstance(key, self._crypto_key_types):
                return key
            key_bytes = force_bytes(key)
            try:
                if key_bytes.startswith(b"ssh-rsa"):
                    return cast(AllowedRSAKeys, load_ssh_public_key(key_bytes))
                if key_bytes.startswith(b"-----BEGIN CERTIFICATE-----"):
                    raise InvalidKeyError(
                        "Could not parse the provided public key."
                    )
                try:
                    return cast(
                        AllowedRSAKeys,
                        load_pem_private_key(key_bytes, password=None),
                    )
                except ValueError:
                    return cast(AllowedRSAKeys, load_pem_public_key(key_bytes))
            except (ValueError, TypeError, UnsupportedAlgorithm):
                raise InvalidKeyError(
                    "Could not parse the provided public key."
                ) from None

        def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
            return key.sign(msg, padding.PKCS1v15(), self.hash_alg())

        def verify(self, msg: bytes, key: AllowedRSAKeys, sig: bytes) -> bool:
            try:
                key.verify(sig, msg, padding.PKCS1v15(), self.hash_alg())
                return True
            except InvalidSignature:
                return False

        @staticmethod
        def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
            if isinstance(key_obj, RSAPrivateKey):
                numbers = key_obj.private_numbers()
                public = numbers.public_numbers
                data: JWKDict = {
                    "kty": "RSA",
                    "key_ops": ["sign"],
                    "n": to_base64url_uint(public.n).decode(),
                    "e": to_base64url_uint(public.e).decode(),
                    "d": to_base64url_uint(numbers.d).decode(),
                    "p": to_base64url_uint(numbers.p).decode(),
                    "q": to_base64url_uint(numbers.q).decode(),
                    "dp": to_base64url_uint(numbers.dmp1).decode(),
                    "dq": to_base64url_uint(numbers.dmq1).decode(),
                    "qi": to_base64url_uint(numbers.iqmp).decode(),
                }
            elif isinstance(key_obj, RSAPublicKey):
                numbers = key_obj.public_numbers()
                data = {
                    "kty": "RSA",
                    "key_ops": ["verify"],
                    "n": to_base64url_uint(numbers.n).decode(),
                    "e": to_base64url_uint(numbers.e).decode(),
                }
            else:
                raise InvalidKeyError("Not a public or private key")
            return data if as_dict else json.dumps(data)

        @staticmethod
        def from_jwk(jwk: str | JWKDict) -> AllowedRSAKeys:
            if isinstance(jwk, str):
                jwk = json.loads(jwk)
            if jwk.get("kty") != "RSA":
                raise InvalidKeyError(
                    "Incorrect key type. Expected: 'RSA', Received: %s" % jwk.get("kty")
                )
            if "n" not in jwk or "e" not in jwk:
                raise InvalidKeyError("Invalid key parameter")

            n = from_base64url_uint(jwk["n"])
            e = from_base64url_uint(jwk["e"])

            if "d" not in jwk:
                return RSAPublicNumbers(e, n).public_key(default_backend())

            d = from_base64url_uint(jwk["d"])
            if "p" in jwk and "q" in jwk:
                p = from_base64url_uint(jwk["p"])
                q = from_base64url_uint(jwk["q"])
            else:
                p, q = rsa_recover_prime_factors(n, e, d)

            dmp1 = from_base64url_uint(jwk["dp"]) if "dp" in jwk else rsa_crt_dmp1(d, p)
            dmq1 = from_base64url_uint(jwk["dq"]) if "dq" in jwk else rsa_crt_dmq1(d, q)
            iqmp = from_base64url_uint(jwk["qi"]) if "qi" in jwk else rsa_crt_iqmp(p, q)

            return RSAPrivateNumbers(
                p=p,
                q=q,
                d=d,
                dmp1=dmp1,
                dmq1=dmq1,
                iqmp=iqmp,
                public_numbers=RSAPublicNumbers(e=e, n=n),
            ).private_key(default_backend())


    class RSAPSSAlgorithm(RSAAlgorithm):
        def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
            return key.sign(
                msg,
                padding.PSS(
                    mgf=padding.MGF1(self.hash_alg()),
                    salt_length=self.hash_alg().digest_size,
                ),
                self.hash_alg(),
            )

        def verify(self, msg: bytes, key: AllowedRSAKeys, sig: bytes) -> bool:
            try:
                key.verify(
                    sig,
                    msg,
                    padding.PSS(
                        mgf=padding.MGF1(self.hash_alg()),
                        salt_length=self.hash_alg().digest_size,
                    ),
                    self.hash_alg(),
                )
                return True
            except InvalidSignature:
                return False


    class ECAlgorithm(Algorithm):
        SHA256: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA256
        SHA384: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA384
        SHA512: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA512

        _crypto_key_types = (EllipticCurvePrivateKey, EllipticCurvePublicKey)

        def __init__(
            self,
            hash_alg: type[hashes.HashAlgorithm],
            curve: type[EllipticCurve],
        ) -> None:
            self.hash_alg = hash_alg
            self.curve = curve

        def prepare_key(self, key: Any) -> AllowedECKeys:
            if isinstance(key, self._crypto_key_types):
                return key
            key_bytes = force_bytes(key)
            try:
                if key_bytes.startswith(b"ecdsa-sha2-"):
                    return cast(AllowedECKeys, load_ssh_public_key(key_bytes))
                try:
                    return cast(
                        AllowedECKeys,
                        load_pem_private_key(key_bytes, password=None),
                    )
                except ValueError:
                    return cast(AllowedECKeys, load_pem_public_key(key_bytes))
            except (ValueError, TypeError, UnsupportedAlgorithm):
                raise InvalidKeyError(
                    "Could not deserialize key data. The data may be in an incorrect format, it may be encrypted with an unsupported algorithm, or it may be an unsupported key type (e.g. EC curves with explicit parameters)."
                ) from None

        def sign(self, msg: bytes, key: EllipticCurvePrivateKey) -> bytes:
            signature = key.sign(msg, ECDSA(self.hash_alg()))
            return der_to_raw_signature(signature, key.curve)

        def verify(self, msg: bytes, key: AllowedECKeys, sig: bytes) -> bool:
            try:
                signature = raw_to_der_signature(sig, key.curve)
                key.verify(signature, msg, ECDSA(self.hash_alg()))
                return True
            except (InvalidSignature, ValueError):
                return False

        @staticmethod
        def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
            if isinstance(key_obj, EllipticCurvePrivateKey):
                numbers = key_obj.private_numbers()
                public_numbers = numbers.public_numbers
                data: JWKDict = {
                    "kty": "EC",
                    "crv": key_obj.curve.name,
                    "x": to_base64url_uint(public_numbers.x).decode(),
                    "y": to_base64url_uint(public_numbers.y).decode(),
                    "d": to_base64url_uint(numbers.private_value).decode(),
                }
            elif isinstance(key_obj, EllipticCurvePublicKey):
                numbers = key_obj.public_numbers()
                data = {
                    "kty": "EC",
                    "crv": key_obj.curve.name,
                    "x": to_base64url_uint(numbers.x).decode(),
                    "y": to_base64url_uint(numbers.y).decode(),
                }
            else:
                raise InvalidKeyError("Not a public or private key")
            return data if as_dict else json.dumps(data)

        @staticmethod
        def from_jwk(jwk: str | JWKDict) -> AllowedECKeys:
            if isinstance(jwk, str):
                jwk = json.loads(jwk)
            if jwk.get("kty") != "EC":
                raise InvalidKeyError(
                    "Incorrect key type. Expected: 'EC', Received: %s" % jwk.get("kty")
                )

            curve_table = {
                "P-256": SECP256R1,
                "P-384": SECP384R1,
                "P-521": SECP521R1,
                "secp256k1": SECP256K1,
            }
            curve_name = jwk.get("crv")
            if curve_name not in curve_table:
                raise InvalidKeyError(
                    "Invalid curve: %s" % curve_name
                )
            if "x" not in jwk or "y" not in jwk:
                raise InvalidKeyError("Invalid key parameter")

            curve = curve_table[curve_name]()
            x = from_base64url_uint(jwk["x"])
            y = from_base64url_uint(jwk["y"])
            public = EllipticCurvePublicNumbers(x, y, curve)

            if "d" not in jwk:
                return public.public_key(default_backend())

            private = EllipticCurvePrivateNumbers(
                from_base64url_uint(jwk["d"]),
                public,
            )
            return private.private_key(default_backend())


    class OKPAlgorithm(Algorithm):
        _crypto_key_types = (
            Ed25519PrivateKey,
            Ed25519PublicKey,
            Ed448PrivateKey,
            Ed448PublicKey,
        )

        def prepare_key(self, key: Any) -> AllowedOKPKeys:
            if isinstance(key, self._crypto_key_types):
                return key
            key_bytes = force_bytes(key)
            try:
                if key_bytes.startswith(b"ssh-"):
                    return cast(AllowedOKPKeys, load_ssh_public_key(key_bytes))
                try:
                    return cast(
                        AllowedOKPKeys,
                        load_pem_private_key(key_bytes, password=None),
                    )
                except ValueError:
                    return cast(AllowedOKPKeys, load_pem_public_key(key_bytes))
            except (ValueError, TypeError, UnsupportedAlgorithm):
                raise InvalidKeyError(
                    "Could not deserialize key data. The data may be in an incorrect format, it may be encrypted with an unsupported algorithm, it may be an unsupported key type, or the key is not in PEM format."
                ) from None

        def sign(self, msg: bytes, key: Any) -> bytes:
            if not isinstance(key, (Ed25519PrivateKey, Ed448PrivateKey)):
                raise InvalidKeyError("Key is not a private key")
            return key.sign(msg)

        def verify(self, msg: bytes, key: Any, sig: bytes) -> bool:
            try:
                if isinstance(key, (Ed25519PrivateKey, Ed448PrivateKey)):
                    key = key.public_key()
                key.verify(sig, msg)
                return True
            except InvalidSignature:
                return False

        @staticmethod
        def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
            if isinstance(key_obj, Ed25519PrivateKey):
                data: JWKDict = {
                    "kty": "OKP",
                    "crv": "Ed25519",
                    "x": base64url_encode(
                        key_obj.public_key().public_bytes(
                            Encoding.Raw,
                            PublicFormat.Raw,
                        )
                    ).decode(),
                    "d": base64url_encode(
                        key_obj.private_bytes(
                            Encoding.Raw,
                            PrivateFormat.Raw,
                            NoEncryption(),
                        )
                    ).decode(),
                }
            elif isinstance(key_obj, Ed448PrivateKey):
                data = {
                    "kty": "OKP",
                    "crv": "Ed448",
                    "x": base64url_encode(
                        key_obj.public_key().public_bytes(
                            Encoding.Raw,
                            PublicFormat.Raw,
                        )
                    ).decode(),
                    "d": base64url_encode(
                        key_obj.private_bytes(
                            Encoding.Raw,
                            PrivateFormat.Raw,
                            NoEncryption(),
                        )
                    ).decode(),
                }
            elif isinstance(key_obj, Ed25519PublicKey):
                data = {
                    "kty": "OKP",
                    "crv": "Ed25519",
                    "x": base64url_encode(
                        key_obj.public_bytes(Encoding.Raw, PublicFormat.Raw)
                    ).decode(),
                }
            elif isinstance(key_obj, Ed448PublicKey):
                data = {
                    "kty": "OKP",
                    "crv": "Ed448",
                    "x": base64url_encode(
                        key_obj.public_bytes(Encoding.Raw, PublicFormat.Raw)
                    ).decode(),
                }
            else:
                raise InvalidKeyError("Not a public or private key")
            return data if as_dict else json.dumps(data)

        @staticmethod
        def from_jwk(jwk: str | JWKDict) -> AllowedOKPKeys:
            if isinstance(jwk, str):
                jwk = json.loads(jwk)
            if jwk.get("kty") != "OKP":
                raise InvalidKeyError(
                    "Incorrect key type. Expected: 'OKP', Received: %s" % jwk.get("kty")
                )

            curve = jwk.get("crv")
            if curve == "Ed25519":
                private_class = Ed25519PrivateKey
                public_class = Ed25519PublicKey
            elif curve == "Ed448":
                private_class = Ed448PrivateKey
                public_class = Ed448PublicKey
            else:
                raise InvalidKeyError("Invalid curve: %s" % curve)

            if "d" in jwk:
                return private_class.from_private_bytes(base64url_decode(jwk["d"]))
            if "x" in jwk:
                return public_class.from_public_bytes(base64url_decode(jwk["x"]))
            raise InvalidKeyError("Invalid key parameter")


def get_default_algorithms() -> dict[str, Algorithm]:
    algorithms: dict[str, Algorithm] = {
        "none": NoneAlgorithm(),
        "HS256": HMACAlgorithm(HMACAlgorithm.SHA256),
        "HS384": HMACAlgorithm(HMACAlgorithm.SHA384),
        "HS512": HMACAlgorithm(HMACAlgorithm.SHA512),
    }
    if has_crypto:
        algorithms.update(
            {
                "RS256": RSAAlgorithm(RSAAlgorithm.SHA256),
                "RS384": RSAAlgorithm(RSAAlgorithm.SHA384),
                "RS512": RSAAlgorithm(RSAAlgorithm.SHA512),
                "ES256": ECAlgorithm(ECAlgorithm.SHA256, SECP256R1),
                "ES256K": ECAlgorithm(ECAlgorithm.SHA256, SECP256K1),
                "ES384": ECAlgorithm(ECAlgorithm.SHA384, SECP384R1),
                "ES521": ECAlgorithm(ECAlgorithm.SHA512, SECP521R1),
                "ES512": ECAlgorithm(ECAlgorithm.SHA512, SECP521R1),
                "PS256": RSAPSSAlgorithm(RSAPSSAlgorithm.SHA256),
                "PS384": RSAPSSAlgorithm(RSAPSSAlgorithm.SHA384),
                "PS512": RSAPSSAlgorithm(RSAPSSAlgorithm.SHA512),
                "EdDSA": OKPAlgorithm(),
            }
        )
    return algorithms