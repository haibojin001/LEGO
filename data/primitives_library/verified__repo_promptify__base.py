from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from promptify.core.config import ModelConfig
from promptify.engine.cost import track_cost
from promptify.engine.llm import LLMEngine
from promptify.parser.parser import Parser
from promptify.prompts.builder import PromptBuilder

logger = logging.getLogger("promptify")


class BaseTask(ABC):
    """Abstract base class for Promptify NLP tasks."""

    def __init__(
        self,
        model: str,
        output_schema: Type[BaseModel],
        instruction: str,
        template: Optional[str] = None,
        domain: Optional[str] = None,
        labels: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        supported_config_keys = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "timeout",
            "max_retries",
        }
        config_values = {
            name: value
            for name, value in kwargs.items()
            if name in supported_config_keys
        }

        config = ModelConfig(
            model=model,
            api_key=api_key,
            **config_values,
        )
        self.engine = LLMEngine(config)
        self.output_schema = output_schema
        self.instruction = instruction
        self.domain = domain
        self.labels = labels
        self.examples = examples
        self.prompt_builder = PromptBuilder(template=template)
        self.parser = Parser()
        self._extra_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in config_values
        }

    def _build_messages(self, text: str, **kwargs: Any) -> List[Dict[str, str]]:
        """Construct the messages sent to the language model."""
        values = {**self._extra_kwargs, **kwargs}
        return self.prompt_builder.build(
            instruction=self.instruction,
            text_input=text,
            domain=self.domain,
            labels=self.labels,
            examples=self.examples,
            output_schema=self.output_schema,
            **values,
        )

    def __call__(self, text: str, **kwargs: Any) -> BaseModel:
        """Run this task synchronously."""
        messages = self._build_messages(text, **kwargs)
        result = self.engine.complete(
            messages,
            output_schema=self.output_schema,
        )
        track_cost(result.cost, result.usage)

        if result.parsed:
            return result.parsed

        return self.parser.parse(result.text, self.output_schema)

    async def acall(self, text: str, **kwargs: Any) -> BaseModel:
        """Run this task asynchronously."""
        messages = self._build_messages(text, **kwargs)
        result = await self.engine.acomplete(
            messages,
            output_schema=self.output_schema,
        )
        track_cost(result.cost, result.usage)

        if result.parsed:
            return result.parsed

        return self.parser.parse(result.text, self.output_schema)

    def batch(
        self,
        texts: List[str],
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> List[BaseModel]:
        """Execute the task for several inputs with limited concurrency."""

        async def run_batch() -> List[BaseModel]:
            limiter = asyncio.Semaphore(max_concurrent)

            async def run_one(item: str) -> BaseModel:
                async with limiter:
                    return await self.acall(item, **kwargs)

            return await asyncio.gather(*(run_one(item) for item in texts))

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop and current_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(run_batch()))
                return future.result()

        return asyncio.run(run_batch())


class Task(BaseTask):
    """General-purpose task using a caller-provided Pydantic output schema."""

    def __init__(
        self,
        model: str,
        output_schema: Type[BaseModel],
        instruction: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=output_schema,
            instruction=instruction,
            **kwargs,
        )