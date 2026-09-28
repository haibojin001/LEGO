from __future__ import annotations


class PyJWTError(Exception):
    """Base class for all exceptions."""


class InvalidTokenError(PyJWTError):
    """Base exception when decode() fails on a token."""


class DecodeError(InvalidTokenError):
    """Raised when a token cannot be decoded because it failed validation."""


class InvalidSignatureError(DecodeError):
    """Raised when a token's signature does not match."""


class ExpiredSignatureError(InvalidTokenError):
    """Raised when a token's exp claim indicates that it has expired."""


class InvalidAudienceError(InvalidTokenError):
    """Raised when a token's aud claim does not match an expected audience."""


class InvalidIssuerError(InvalidTokenError):
    """Raised when a token's iss claim does not match the expected issuer."""


class InvalidIssuedAtError(InvalidTokenError):
    """Raised when a token's iat claim is non-numeric."""


class ImmatureSignatureError(InvalidTokenError):
    """Raised when a token's nbf or iat claim represents a future time."""


class InvalidKeyError(PyJWTError):
    """Raised when the specified key is not in the proper format."""


class InvalidAlgorithmError(InvalidTokenError):
    """Raised when the specified algorithm is not recognized."""


class MissingRequiredClaimError(InvalidTokenError):
    """Raised when a required claim is absent from the claimset."""

    def __init__(self, claim: str) -> None:
        self.claim = claim

    def __str__(self) -> str:
        return f'Token is missing the "{self.claim}" claim'


class PyJWKError(PyJWTError):
    pass


class MissingCryptographyError(PyJWKError):
    """Raised when cryptography is required but unavailable."""


class PyJWKSetError(PyJWTError):
    pass


class PyJWKClientError(PyJWKError):
    pass


class PyJWKClientConnectionError(PyJWKClientError):
    pass


class InvalidSubjectError(InvalidTokenError):
    """Raised when a token's sub claim is invalid."""


class InvalidJTIError(InvalidTokenError):
    """Raised when a token's jti claim is not a string."""