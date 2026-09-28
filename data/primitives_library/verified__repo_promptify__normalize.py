from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from promptify.tasks.base import BaseTask


class _NormalizationResult(BaseModel):
    normalized_text: str


class NormalizeText(BaseTask):
    """Task for normalizing text using provided rules."""

    def __init__(
        self,
        model: str,
        rules: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if instruction is None:
            instruction = "Normalize the text according to the rules and return JSON."

        super().__init__(
            model=model,
            output_schema=_NormalizationResult,
            instruction=instruction,
            template="text_normalization",
            examples=examples,
            rules=rules,
            **kwargs,
        )


class _Topic(BaseModel):
    topic: str
    words: List[str]


class _TopicResult(BaseModel):
    topics: List[_Topic]


class ExtractTopics(BaseTask):
    """Task for extracting coherent topics from text."""

    def __init__(
        self,
        model: str,
        num_topics: int = 5,
        domain: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if instruction is None:
            instruction = "Extract coherent topics from the text and return JSON."

        super().__init__(
            model=model,
            output_schema=_TopicResult,
            instruction=instruction,
            template="topic_modelling",
            domain=domain,
            num_topics=num_topics,
            **kwargs,
        )