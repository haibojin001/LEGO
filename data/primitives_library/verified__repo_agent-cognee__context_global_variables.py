import os
import inspect
import warnings
from contextvars import ContextVar
from typing import Optional
from uuid import UUID

from cognee.base_config import get_base_config
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.databases.vector.config import (
    get_vectordb_config,
    get_vectordb_context_config,
)
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.modules.users.methods import get_user
from cognee.infrastructure.databases.graph.config import (
    get_graph_config,
    get_graph_context_config,
)
from cognee.infrastructure.databases.utils.get_or_create_dataset_database import (
    get_or_create_dataset_database,
)
from cognee.infrastructure.databases.utils.resolve_dataset_database_connection_info import (
    resolve_dataset_database_connection_info,
)


vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)
current_dataset_id: ContextVar[Optional[UUID]] = ContextVar("current_dataset_id", default=None)
llm_config: ContextVar[Optional[LLMConfig]] = ContextVar("llm_config", default=None)
embedding_config = ContextVar("embedding_config", default=None)
session_user = ContextVar("session_user", default=None)
current_pipeline_stage: ContextVar[Optional[str]] = ContextVar(
    "current_pipeline_stage", default=None
)


async def set_session_user_context_variable(user):
    session_user.set(user)


def multi_user_support_possible():
    graph_configuration = get_graph_config()
    vector_configuration = get_vectordb_config()

    graph_handler = graph_configuration.graph_dataset_database_handler
    vector_handler = vector_configuration.vector_dataset_database_handler

    from cognee.infrastructure.databases.dataset_database_handler import (
        supported_dataset_database_handlers,
    )

    if graph_handler not in supported_dataset_database_handlers:
        raise EnvironmentError(
            "Unsupported graph dataset to database handler configured. Cannot add support for multi-user access control mode. Please use a supported graph dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected graph dataset to database handler: {graph_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    if vector_handler not in supported_dataset_database_handlers:
        raise EnvironmentError(
            "Unsupported vector dataset to database handler configured. Cannot add support for multi-user access control mode. Please use a supported vector dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected vector dataset to database handler: {vector_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    def providers_for(handler):
        provider = supported_dataset_database_handlers[handler]["handler_provider"]
        if isinstance(provider, str):
            return (provider,)
        return provider

    if graph_configuration.graph_database_provider not in providers_for(graph_handler):
        raise EnvironmentError(
            "The selected graph dataset to database handler does not work with the configured graph database provider. Cannot add support for multi-user access control mode. Please use a supported graph dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected graph database provider: {graph_configuration.graph_database_provider}\n"
            f"Selected graph dataset to database handler: {graph_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    if vector_configuration.vector_db_provider not in providers_for(vector_handler):
        raise EnvironmentError(
            "The selected vector dataset to database handler does not work with the configured vector database provider. Cannot add support for multi-user access control mode. Please use a supported vector dataset to database handler or set the environment variables ENABLE_BACKEND_ACCESS_CONTROL to false to switch off multi-user access control mode.\n"
            f"Selected vector database provider: {vector_configuration.vector_db_provider}\n"
            f"Selected vector dataset to database handler: {vector_handler}\n"
            f"Supported dataset to database handlers: {list(supported_dataset_database_handlers.keys())}\n"
        )

    return True


def backend_access_control_enabled():
    configured_value = os.environ.get("ENABLE_BACKEND_ACCESS_CONTROL", None)

    if configured_value is None:
        return multi_user_support_possible()

    if configured_value.lower() == "true":
        return multi_user_support_possible()

    return False


VECTOR_DBS_WITH_MULTI_USER_SUPPORT = ["lancedb", "pgvector", "falkor"]
GRAPH_DBS_WITH_MULTI_USER_SUPPORT = ["ladybug", "kuzu", "falkor", "postgres"]


async def _await_if_needed(value):
    if inspect.isawaitable(value):
        return await value
    return value


class DatabaseContextManager:
    __slots__ = (
        "_dataset",
        "_user_id",
        "_llm_config",
        "_embedding_config",
        "_applied",
        "_dataset_token",
        "_llm_token",
        "_embedding_token",
    )

    def __init__(
        self,
        dataset: Optional[UUID],
        user_id: UUID,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> None:
        self._dataset = dataset
        self._user_id = user_id
        self._llm_config = llm_config
        self._embedding_config = embedding_config
        self._applied = False
        self._dataset_token = None
        self._llm_token = None
        self._embedding_token = None

    async def apply_database_context_variables(
        self, dataset: Optional[UUID], user_id: UUID
    ) -> None:
        if dataset is not None and not isinstance(dataset, UUID):
            raise CogneeValidationError(
                message=f"dataset must be a dataset id (UUID), got {dataset!r}. "
                "Resolve dataset names to ids before entering the database context."
            )

        self._dataset_token = current_dataset_id.set(dataset)

        if self._llm_config is not None:
            self._llm_token = llm_config.set(self._llm_config)

        if self._embedding_config is not None:
            self._embedding_token = embedding_config.set(self._embedding_config)

        if not backend_access_control_enabled():
            return

        if dataset is None:
            raise CogneeValidationError(
                "A dataset must be provided when backend access control is enabled."
            )

        user = await get_user(user_id)

        if user is None:
            warnings.warn(
                f"Unable to find user {user_id!r} while setting database context.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        dataset_database = await _await_if_needed(
            get_or_create_dataset_database(dataset, user)
        )
        connection_info = await _await_if_needed(
            resolve_dataset_database_connection_info(dataset_database)
        )

        graph_context = await _await_if_needed(
            get_graph_context_config(connection_info)
        )
        vector_context = await _await_if_needed(
            get_vectordb_context_config(connection_info)
        )

        graph_db_config.set(graph_context)
        vector_db_config.set(vector_context)

    async def __aenter__(self):
        if self._applied:
            raise RuntimeError(
                "DatabaseContextManager instances are single-use and cannot be entered twice."
            )

        await self.apply_database_context_variables(self._dataset, self._user_id)
        self._applied = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._embedding_token is not None:
            embedding_config.reset(self._embedding_token)

        if self._llm_token is not None:
            llm_config.reset(self._llm_token)

        if self._dataset_token is not None:
            current_dataset_id.reset(self._dataset_token)

        return False

    def __await__(self):
        async def apply():
            if self._applied:
                raise RuntimeError(
                    "DatabaseContextManager instances are single-use and cannot be awaited twice."
                )

            await self.apply_database_context_variables(self._dataset, self._user_id)
            self._applied = True

        return apply().__await__()


def set_database_global_context_variables(
    dataset: Optional[UUID],
    user_id: UUID,
    llm_config: Optional[LLMConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
) -> DatabaseContextManager:
    return DatabaseContextManager(
        dataset=dataset,
        user_id=user_id,
        llm_config=llm_config,
        embedding_config=embedding_config,
    )