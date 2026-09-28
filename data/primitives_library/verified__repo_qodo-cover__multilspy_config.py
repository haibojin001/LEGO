from dataclasses import dataclass
from enum import Enum
import inspect


class Language(str, Enum):
    CSHARP = "csharp"
    PYTHON = "python"
    RUST = "rust"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"

    def __str__(self) -> str:
        return self.value


@dataclass
class MultilspyConfig:
    code_language: Language
    trace_lsp_communication: bool = False

    @classmethod
    def from_dict(cls, env: dict):
        return cls(
            **{
                key: value
                for key, value in env.items()
                if key in inspect.signature(cls).parameters
            }
        )