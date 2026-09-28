from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import litellm
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from promptify.core.config import ModelConfig
from promptify.core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelResponseError,
)

logger = logging.getLogger("promptify")


@dataclass
class LLMResponse:
    """Response returned by an LLM engine."""

    text: str
    parsed: Optional[BaseModel] = None
    raw_response: Any = None
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    cost: float = 0.0


class LLMEngine:
    """Provider-independent LLM client implemented through LiteLLM."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        litellm.drop_params = True

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
        }

        if self.config.api_key:
            request["api_key"] = self.config.api_key
        if self.config.max_tokens:
            request["max_tokens"] = self.config.max_tokens
        if self.config.stop:
            request["stop"] = self.config.stop
        if self.config.timeout:
            request["timeout"] = self.config.timeout

        if output_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                    "strict": False,
                },
            }

        request.update(self.config.extra_params)
        request.update(kwargs)
        return request

    def _map_exception(self, exc: Exception) -> Exception:
        message = str(exc).lower()

        if "auth" in message or "api key" in message or "401" in message:
            return ModelAuthenticationError(str(exc))
        if "rate" in message or "429" in message:
            return ModelRateLimitError(str(exc))
        if "connect" in message or "timeout" in message:
            return ModelConnectionError(str(exc))

        return ModelResponseError(str(exc))

    def _parse_response(
        self,
        response: Any,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        selected_choice = response.choices[0]
        content = selected_choice.message.content or ""

        token_usage: Dict[str, int] = {}
        if hasattr(response, "usage") and response.usage:
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        response_cost = 0.0
        try:
            response_cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        structured_result = None
        if output_schema and content:
            try:
                structured_result = output_schema.model_validate(json.loads(content))
            except Exception:
                logger.debug(
                    "Structured parse failed, raw text available in response"
                )

        return LLMResponse(
            text=content,
            parsed=structured_result,
            raw_response=response,
            usage=token_usage,
            model=response.model or self.config.model,
            cost=response_cost,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a synchronous chat completion."""
        parameters = self._build_params(messages, output_schema, **kwargs)

        @retry(
            retry=retry_if_exception_type(
                (ModelConnectionError, ModelRateLimitError)
            ),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self.config.max_retries),
            reraise=True,
        )
        def invoke() -> LLMResponse:
            try:
                result = litellm.completion(**parameters)
                return self._parse_response(result, output_schema)
            except Exception as exc:
                mapped_error = self._map_exception(exc)
                if isinstance(
                    mapped_error,
                    (ModelConnectionError, ModelRateLimitError),
                ):
                    raise mapped_error from exc
                raise mapped_error from exc

        return invoke()

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run an asynchronous chat completion."""
        parameters = self._build_params(messages, output_schema, **kwargs)

        try:
            result = await litellm.acompletion(**parameters)
            return self._parse_response(result, output_schema)
        except Exception as exc:
            raise self._map_exception(exc) from exc