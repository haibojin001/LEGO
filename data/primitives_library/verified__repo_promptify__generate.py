from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from promptify.schemas.generate import GeneratedQuestion, SQLQuery
from promptify.tasks.base import BaseTask


class _QuestionGenResult(BaseModel):
    questions: List[GeneratedQuestion]


class GenerateQuestions(BaseTask):
    """Generate question-answer pairs from supplied text."""

    def __init__(
        self,
        model: str,
        num_questions: int = 3,
        domain: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolved_instruction = instruction
        if resolved_instruction is None:
            resolved_instruction = (
                "Generate question-answer pairs from the text and return JSON."
            )

        super().__init__(
            model=model,
            output_schema=_QuestionGenResult,
            instruction=resolved_instruction,
            template="question_generation",
            domain=domain,
            num_questions=num_questions,
            **kwargs,
        )


class GenerateSQL(BaseTask):
    """Translate natural-language requests into SQL queries."""

    def __init__(
        self,
        model: str,
        schema: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolved_instruction = instruction
        if resolved_instruction is None:
            resolved_instruction = (
                "Convert the natural language query to SQL and return JSON."
            )

        super().__init__(
            model=model,
            output_schema=SQLQuery,
            instruction=resolved_instruction,
            template="sql_writer",
            examples=examples,
            schema=schema,
            **kwargs,
        )