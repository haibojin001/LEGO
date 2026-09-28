import os

import backoff
import numpy as np
from anthropic import Anthropic
from openai import (
    APIConnectionError,
    APIError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)
from google import genai
from google.genai import types


class LMMEngine:
    pass


class OpenAIEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key=None,
    ):
        self.model = embedding_model
        self.api_key = api_key

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )

        client = OpenAI(api_key=key)
        result = client.embeddings.create(model=self.model, input=text)
        return np.array([item.embedding for item in result.data])


class GeminiEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        embedding_model: str = "text-embedding-004",
        api_key=None,
    ):
        self.model = embedding_model
        self.api_key = api_key

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )

        client = genai.Client(api_key=key)
        result = client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return np.array([embedding.values for embedding in result.embeddings])


class AzureOpenAIEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key=None,
        api_version=None,
        endpoint_url=None,
    ):
        self.model = embedding_model
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint_url = endpoint_url

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )

        version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if version is None:
            raise ValueError(
                "An API Version needs to be provided in either the api_version parameter or as an environment variable named OPENAI_API_VERSION"
            )

        endpoint = self.endpoint_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint is None:
            raise ValueError(
                "An Endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )

        client = AzureOpenAI(
            api_key=key,
            api_version=version,
            azure_endpoint=endpoint,
        )
        result = client.embeddings.create(input=text, model=self.model)
        return np.array([item.embedding for item in result.data])


class LMMEngineOpenAI(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APIError, RateLimitError),
        max_time=60,
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )

        if not self.llm_client:
            if self.base_url:
                self.llm_client = OpenAI(base_url=self.base_url, api_key=key)
            else:
                self.llm_client = OpenAI(api_key=key)

        result = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        return result.choices[0].message.content


class LMMEngineAnthropic(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, thinking=False, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APIError, RateLimitError),
        max_time=60,
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )

        if not self.llm_client:
            self.llm_client = Anthropic(api_key=key)

        if self.thinking:
            result = self.llm_client.messages.create(
                system=messages[0]["content"][0]["text"],
                model=self.model,
                messages=messages[1:],
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **kwargs,
            )
            thoughts = result.content[0].thinking
            print("CLAUDE 3.7 THOUGHTS:", thoughts)
            return result.content[1].text

        result = self.llm_client.messages.create(
            system=messages[0]["content"][0]["text"],
            model=self.model,
            messages=messages[1:],
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        return result.content[0].text


class LMMEngineGemini(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APIError, RateLimitError),
        max_time=60,
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )

        base_url = self.base_url or os.getenv("GEMINI_ENDPOINT_URL")

        if not self.llm_client:
            if base_url:
                self.llm_client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(base_url=base_url),
                )
            else:
                self.llm_client = genai.Client(api_key=key)

        result = self.llm_client.models.generate_content(
            model=self.model,
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens if max_new_tokens else 4096,
                **kwargs,
            ),
        )
        return result.text