import base64
import os

import numpy as np

from gui_agents.s3.core.engine import (
    LMMEngineAnthropic,
    LMMEngineAzureOpenAI,
    LMMEngineGemini,
    LMMEngineHuggingFace,
    LMMEngineOpenAI,
    LMMEngineOpenRouter,
    LMMEngineParasail,
    LMMEnginevLLM,
)


class LMMAgent:
    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get("engine_type")

                if engine_type == "openai":
                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "anthropic":
                    self.engine = LMMEngineAnthropic(**engine_params)
                elif engine_type == "azure":
                    self.engine = LMMEngineAzureOpenAI(**engine_params)
                elif engine_type == "vllm":
                    self.engine = LMMEnginevLLM(**engine_params)
                elif engine_type == "huggingface":
                    self.engine = LMMEngineHuggingFace(**engine_params)
                elif engine_type == "gemini":
                    self.engine = LMMEngineGemini(**engine_params)
                elif engine_type == "open_router":
                    self.engine = LMMEngineOpenRouter(**engine_params)
                elif engine_type == "parasail":
                    self.engine = LMMEngineParasail(**engine_params)
                elif engine_type == "ollama":
                    if not engine_params.get("base_url"):
                        base_url = os.getenv("OLLAMA_HOST")
                        if base_url:
                            if not base_url.endswith("/v1"):
                                base_url = base_url.rstrip("/") + "/v1"
                            engine_params["base_url"] = base_url
                        else:
                            raise ValueError(
                                "Ollama endpoint must be provided via 'base_url' parameter or 'OLLAMA_HOST' environment variable."
                            )

                    if not engine_params.get("api_key"):
                        engine_params["api_key"] = "ollama"

                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "deepseek":
                    if "base_url" not in engine_params:
                        base_url = os.getenv("DEEPSEEK_ENDPOINT_URL")
                        if not base_url:
                            base_url = "https://api.deepseek.com"
                        if not base_url.endswith("/v1"):
                            base_url = base_url.rstrip("/") + "/v1"
                        engine_params["base_url"] = base_url

                    if not engine_params.get("api_key"):
                        api_key = os.getenv("DEEPSEEK_API_KEY")
                        if not api_key:
                            raise ValueError(
                                "DeepSeek API key must be provided via 'api_key' parameter or 'DEEPSEEK_API_KEY' environment variable."
                            )
                        engine_params["api_key"] = api_key

                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "qwen":
                    if not engine_params.get("base_url"):
                        base_url = os.getenv("QWEN_ENDPOINT_URL")
                        if not base_url:
                            base_url = (
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"
                            )
                        if not base_url.endswith("/v1"):
                            base_url = base_url.rstrip("/") + "/v1"
                        engine_params["base_url"] = base_url

                    if not engine_params.get("api_key"):
                        api_key = os.getenv("QWEN_API_KEY")
                        if not api_key:
                            raise ValueError(
                                "Qwen API key must be provided via 'api_key' parameter or 'QWEN_API_KEY' environment variable."
                            )
                        engine_params["api_key"] = api_key

                    self.engine = LMMEngineOpenAI(**engine_params)
                else:
                    raise ValueError(f"engine_type '{engine_type}' is not supported")
            else:
                raise ValueError("engine_params must be provided")
        else:
            self.engine = engine

        self.messages = []

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    def encode_image(self, image_content):
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        return base64.b64encode(image_content).decode("utf-8")

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        message = {
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt}],
        }

        if len(self.messages) > 0:
            self.messages[0] = message
        else:
            self.messages.append(message)

    def remove_message_at(self, index):
        """Remove a message at a given index"""
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(
        self, index, text_content, image_content=None, image_detail="high"
    ):
        """Replace a message at a given index"""
        if index < len(self.messages):
            self.messages[index] = {
                "role": self.messages[index]["role"],
                "content": [{"type": "text", "text": text_content}],
            }

            if image_content:
                base64_image = self.encode_image(image_content)
                self.messages[index]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

    def add_message(
        self,
        text_content,
        image_content=None,
        role=None,
        image_detail="high",
        put_text_last=False,
    ):
        """Add a new message to the list of messages"""
        if isinstance(
            self.engine,
            (
                LMMEngineOpenAI,
                LMMEngineAzureOpenAI,
                LMMEngineHuggingFace,
                LMMEngineGemini,
                LMMEngineOpenRouter,
                LMMEngineParasail,
            ),
        ):
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if isinstance(image_content, np.ndarray) or image_content:
                if isinstance(image_content, list):
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            }
                        )
                else:
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": image_detail,
                            },
                        }
                    )

            if put_text_last:
                message["content"] = (
                    message["content"][1:] + message["content"][:1]
                )

            self.messages.append(message)

        elif isinstance(self.engine, LMMEngineAnthropic):
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if isinstance(image_content, np.ndarray) or image_content:
                images = image_content if isinstance(image_content, list) else [image_content]
                for image in images:
                    base64_image = self.encode_image(image)
                    message["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }
                    )

            if put_text_last:
                message["content"] = (
                    message["content"][1:] + message["content"][:1]
                )

            self.messages.append(message)

        elif isinstance(self.engine, LMMEnginevLLM):
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if isinstance(image_content, np.ndarray) or image_content:
                images = image_content if isinstance(image_content, list) else [image_content]
                for image in images:
                    base64_image = self.encode_image(image)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": image_detail,
                            },
                        }
                    )

            if put_text_last:
                message["content"] = (
                    message["content"][1:] + message["content"][:1]
                )

            self.messages.append(message)

    def get_response(self):
        return self.engine.generate(self.messages)