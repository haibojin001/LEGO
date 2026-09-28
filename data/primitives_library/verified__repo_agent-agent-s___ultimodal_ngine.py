import os
import re
from io import BytesIO

import backoff
import numpy as np
import openai
import requests
from anthropic import Anthropic
from openai import APIConnectionError, APIError, AzureOpenAI, OpenAI, RateLimitError
from PIL import Image


def image_parser(args):
    return args.image_file.split(args.sep)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def load_images(image_files):
    images = []
    for image_file in image_files:
        images.append(load_image(image_file))
    return images


class LMMEngine:
    pass


class LMMEngineOpenAI(LMMEngine):
    def __init__(self, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )

        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = OpenAI(api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APIError, RateLimitError),
        max_time=60,
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        return completion.choices[0].message.content


class LMMEngineAnthropic(LMMEngine):
    def __init__(self, api_key=None, model=None, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )

        self.api_key = api_key
        self.llm_client = Anthropic(api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APIError, RateLimitError),
        max_time=60,
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        response = self.llm_client.messages.create(
            system=messages[0]["content"][0]["text"],
            model=self.model,
            messages=messages[1:],
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        return response.content[0].text


class OpenAIEmbeddingEngine(LMMEngine):
    def __init__(
        self,
        api_key=None,
        rate_limit: int = -1,
        display_cost: bool = True,
    ):
        """Init an OpenAI Embedding engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            display_cost (bool, optional): Display cost of API call. Defaults to True.
        """
        self.model = "text-embedding-3-small"
        self.cost_per_thousand_tokens = 0.00002

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )

        self.api_key = api_key
        self.display_cost = display_cost
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def get_embeddings(self, text: str) -> np.ndarray:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=text)

        if self.display_cost:
            total_tokens = response.usage.total_tokens
            cost = self.cost_per_thousand_tokens * total_tokens / 1000

        return np.array([item.embedding for item in response.data])


class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(
        self,
        api_key=None,
        azure_endpoint=None,
        model=None,
        api_version=None,
        rate_limit=-1,
        **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model

        assert api_version is not None, "api_version must be provided"
        self.api_version = api_version

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )

        self.api_key = api_key

        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint is None:
            raise ValueError(
                "An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_API_BASE"
            )

        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        self.cost = 0.0

    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        total_tokens = completion.usage.total_tokens
        self.cost += 0.02 * ((total_tokens + 500) / 1000)
        return completion.choices[0].message.content


class LMMEnginevLLM(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key

        self.base_url = base_url or os.getenv("vLLM_ENDPOINT_URL")
        if self.base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL"
            )

        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs
    ):
        """Generate the next message based on previous messages"""
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
            **kwargs,
        )
        return completion.choices[0].message.content