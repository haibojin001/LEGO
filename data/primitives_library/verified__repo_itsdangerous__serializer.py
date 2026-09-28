from __future__ import annotations

import collections.abc as cabc
import json
import typing as t

from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import Signer
from .signer import _make_keys_list

if t.TYPE_CHECKING:
    import typing_extensions as te

    _TSerialized = te.TypeVar("_TSerialized", bound=str | bytes, default=str | bytes)
else:
    _TSerialized = t.TypeVar("_TSerialized", bound=str | bytes)


class _PDataSerializer(t.Protocol[_TSerialized]):
    def loads(self, payload: _TSerialized, /) -> t.Any: ...
    def dumps(self, obj: t.Any, /) -> _TSerialized: ...


def is_text_serializer(
    serializer: _PDataSerializer[t.Any],
) -> te.TypeGuard[_PDataSerializer[str]]:
    """Return whether the serializer emits string data."""
    return isinstance(serializer.dumps({}), str)


class Serializer(t.Generic[_TSerialized]):
    """Serialize objects, sign their representation, and verify it later."""

    default_serializer: _PDataSerializer[t.Any] = json
    default_signer: type[Signer] = Signer
    default_fallback_signers: list[
        dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
    ] = []

    @t.overload
    def __init__(
        self: Serializer[str],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: None | _PDataSerializer[str] = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: t.Any | None = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None:
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)

        if salt is not None:
            salt = want_bytes(salt)

        self.salt = salt

        if serializer is None:
            serializer = self.default_serializer

        self.serializer: _PDataSerializer[_TSerialized] = serializer
        self.is_text_serializer: bool = is_text_serializer(serializer)
        self.serializer_kwargs: dict[str, t.Any] = serializer_kwargs or {}

        if signer is None:
            signer = self.default_signer

        self.signer: type[Signer] = signer
        self.signer_kwargs: dict[str, t.Any] = signer_kwargs or {}

        if fallback_signers is None:
            fallback_signers = self.default_fallback_signers

        self.fallback_signers = fallback_signers

    @property
    def secret_key(self) -> bytes:
        """The most recently configured secret key."""
        return self.secret_keys[-1]

    def make_signer(self, salt: str | bytes | None = None) -> Signer:
        """Create the signer used for newly generated signatures."""
        if salt is None:
            salt = self.salt

        return self.signer(self.secret_keys, salt=salt, **self.signer_kwargs)

    def iter_unsigners(
        self, salt: str | bytes | None = None
    ) -> cabc.Iterator[Signer]:
        """Yield all signers accepted when validating a value."""
        if salt is None:
            salt = self.salt

        yield self.make_signer(salt)

        for item in self.fallback_signers:
            if isinstance(item, dict):
                signer_class = self.signer
                options = item
            elif isinstance(item, tuple):
                signer_class, options = item
            else:
                signer_class = item
                options = {}

            yield signer_class(self.secret_keys, salt=salt, **options)

    def dump_payload(self, obj: t.Any) -> bytes:
        """Convert an object to its unsigned byte representation."""
        return want_bytes(self.serializer.dumps(obj, **self.serializer_kwargs))

    def dumps(
        self, obj: t.Any, salt: str | bytes | None = None
    ) -> _TSerialized:
        """Serialize and sign an object."""
        signed = self.make_signer(salt).sign(self.dump_payload(obj))

        if self.is_text_serializer:
            return t.cast(_TSerialized, signed.decode("utf-8"))

        return t.cast(_TSerialized, signed)

    def load_payload(
        self,
        payload: bytes,
        serializer: _PDataSerializer[t.Any] | None = None,
    ) -> t.Any:
        """Deserialize an unsigned payload."""
        if serializer is None:
            serializer = self.serializer

        try:
            if self.is_text_serializer:
                payload = payload.decode("utf-8")

            return serializer.loads(payload)
        except Exception as error:
            raise BadPayload(
                "Could not load the payload because an exception occurred"
                " on unserializing the data.",
                original_error=error,
            ) from error

    def loads(
        self,
        s: str | bytes,
        salt: str | bytes | None = None,
        **kwargs: t.Any,
    ) -> t.Any:
        """Verify a signed value and deserialize its payload."""
        signed = want_bytes(s)
        failure: BadSignature | None = None

        for signer in self.iter_unsigners(salt):
            try:
                payload = signer.unsign(signed)
            except BadSignature as error:
                failure = error
                continue

            return self.load_payload(payload, **kwargs)

        raise t.cast(BadSignature, failure)

    def loads_unsafe(
        self,
        s: str | bytes,
        salt: str | bytes | None = None,
        **kwargs: t.Any,
    ) -> tuple[bool, t.Any]:
        """Load data even when its signature cannot be verified."""
        try:
            return True, self.loads(s, salt=salt, **kwargs)
        except BadSignature as error:
            if error.payload is None:
                return False, None

            try:
                return False, self.load_payload(error.payload, **kwargs)
            except BadPayload:
                return False, None