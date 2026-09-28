from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import ErrorMapping


class EnvError(ValueError):
    """Base exception for errors raised while parsing environment variables."""


class EnvNotSetError(EnvError):
    """Raised when a required environment variable is unset."""


class EnvValidationError(EnvError):
    """Raised when validation fails against one or all parsed environment variables."""

    def __init__(self, message: str, error_messages: list[str] | ErrorMapping) -> None:
        self.error_messages = error_messages
        super().__init__(message)


class EnvSealedError(TypeError, EnvError):
    """Raised when parsing new values after an Env instance has been sealed."""


class ParserConflictError(ValueError):
    """Raised when a custom parser conflicts with a built-in parser method."""