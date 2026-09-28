from typing import Any

from ...types import Extension
from .dataloaders import LoaderRegistry


class SQLAlchemyDataLoaderExtension(Extension):
    """Ariadne extension that creates a per-request LoaderRegistry."""

    def __init__(
        self,
        *,
        session_key: str = "session",
        registry_key: str = "loader_registry",
    ):
        self.session_key = session_key
        self.registry_key = registry_key

    def request_started(self, context: Any) -> None:
        context[self.registry_key] = LoaderRegistry(context[self.session_key])