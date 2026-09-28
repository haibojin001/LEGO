import os
import base64
from pathlib import Path
from typing import Optional
from functools import lru_cache

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.root_dir import get_absolute_path, ensure_absolute_path
from cognee.modules.observability.observers import Observer
from cognee.shared.logging_utils import get_logger


logger = get_logger()


def _tracing_explicitly_disabled() -> bool:
    value = os.getenv("COGNEE_TRACING_ENABLED", "")
    return value.lower() in ("false", "0", "no")


class BaseConfig(BaseSettings):
    data_root_directory: str = get_absolute_path(".data_storage")
    system_root_directory: str = get_absolute_path(".cognee_system")
    cache_root_directory: str = get_absolute_path(".cognee_cache")
    logs_root_directory: str = os.getenv(
        "COGNEE_LOGS_DIR",
        str(Path.home() / ".cognee" / "logs"),
    )
    monitoring_tool: object = Observer.NONE

    default_feedback_influence: float = float(
        os.getenv("DEFAULT_FEEDBACK_INFLUENCE", "0.0")
    )
    preference_alpha: float = float(os.getenv("PREFERENCE_ALPHA", "0.3"))
    preference_beta: float = float(os.getenv("PREFERENCE_BETA", "0.02"))
    personalization_enabled: bool = os.getenv(
        "PERSONALIZATION_ENABLED",
        "false",
    ).lower() in ("true", "1", "yes")
    personalization_influence: float = float(
        os.getenv("PERSONALIZATION_INFLUENCE", "0.3")
    )

    default_user_email: Optional[str] = os.getenv("DEFAULT_USER_EMAIL")
    default_user_password: Optional[str] = os.getenv("DEFAULT_USER_PASSWORD")

    cognee_tracing_enabled: bool = os.getenv(
        "COGNEE_TRACING_ENABLED",
        "false",
    ).lower() in ("true", "1", "yes")
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "cognee")
    otel_exporter_otlp_endpoint: Optional[str] = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_exporter_otlp_headers: Optional[str] = os.getenv(
        "OTEL_EXPORTER_OTLP_HEADERS"
    )

    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @pydantic.model_validator(mode="after")
    def validate_personalization_knobs(self):
        if not 0.0 < self.preference_alpha <= 1.0:
            raise ValueError(
                f"PREFERENCE_ALPHA must be in (0, 1], got {self.preference_alpha}"
            )
        if not 0.0 <= self.preference_beta < 1.0:
            raise ValueError(
                f"PREFERENCE_BETA must be in [0, 1), got {self.preference_beta}"
            )
        if not 0.0 <= self.personalization_influence <= 1.0:
            raise ValueError(
                "PERSONALIZATION_INFLUENCE must be in [0, 1], "
                f"got {self.personalization_influence}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_paths(self):
        backend = os.getenv("STORAGE_BACKEND", "").lower()
        configured_cache_root = os.getenv("CACHE_ROOT_DIRECTORY")

        if backend == "s3" and not configured_cache_root:
            bucket = os.getenv("STORAGE_BUCKET_NAME")
            if bucket:
                self.cache_root_directory = f"s3://{bucket}/cognee/cache"

        self.data_root_directory = ensure_absolute_path(self.data_root_directory)
        self.system_root_directory = ensure_absolute_path(self.system_root_directory)
        self.cache_root_directory = ensure_absolute_path(self.cache_root_directory)
        self.logs_root_directory = ensure_absolute_path(self.logs_root_directory)

        if self.langfuse_public_key or self.langfuse_secret_key:
            if not (self.langfuse_public_key and self.langfuse_secret_key):
                raise ValueError(
                    "Both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be provided together."
                )

            credentials = (
                f"{self.langfuse_public_key}:{self.langfuse_secret_key}"
            ).encode("utf-8")
            encoded_credentials = base64.b64encode(credentials).decode("utf-8")

            host = (
                self.langfuse_host
                or os.getenv("LANGFUSE_BASE_URL")
                or "https://cloud.langfuse.com"
            )

            if not self.otel_exporter_otlp_endpoint:
                self.otel_exporter_otlp_endpoint = (
                    f"{host.rstrip('/')}/api/public/otel/v1/traces"
                )

            if not self.otel_exporter_otlp_headers:
                self.otel_exporter_otlp_headers = (
                    f"Authorization=Basic {encoded_credentials}"
                )

            if not _tracing_explicitly_disabled():
                self.cognee_tracing_enabled = True

        if self.default_feedback_influence > 0:
            logger.warning(
                "DEFAULT_FEEDBACK_INFLUENCE=%s defers every hybrid search to graph completion",
                self.default_feedback_influence,
            )

        return self

    def to_dict(self) -> dict:
        return {
            "data_root_directory": self.data_root_directory,
            "system_root_directory": self.system_root_directory,
            "monitoring_tool": self.monitoring_tool,
            "cache_root_directory": self.cache_root_directory,
            "logs_root_directory": self.logs_root_directory,
        }


@lru_cache
def get_base_config():
    return BaseConfig()