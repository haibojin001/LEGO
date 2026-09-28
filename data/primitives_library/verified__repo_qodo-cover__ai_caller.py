import datetime
import os
import time
from functools import wraps
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_fixed
from wandb.sdk.data_types.trace_tree import Trace

from cover_agent.custom_logger import CustomLogger
from cover_agent.record_replay_manager import RecordReplayManager
from cover_agent.settings.config_loader import get_settings
from cover_agent.utils import get_original_caller


def conditional_retry(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if not self.enable_retry:
            return function(self, *args, **kwargs)

        retry_count = get_settings().get("default").get("model_retries", 3)

        @retry(stop=stop_after_attempt(retry_count), wait=wait_fixed(1))
        def invoke():
            return function(self, *args, **kwargs)

        return invoke()

    return wrapped


class AICaller:
    def __init__(
        self,
        model: str,
        api_base: str = "",
        enable_retry=True,
        max_tokens=16384,
        source_file: str = None,
        test_file: str = None,
        record_mode: bool = False,
        record_replay_manager: Optional[RecordReplayManager] = None,
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        self.model = model
        self.api_base = api_base
        self.enable_retry = enable_retry
        self.max_tokens = max_tokens
        self.source_file = source_file
        self.test_file = test_file
        self.record_mode = record_mode
        self.record_replay_manager = record_replay_manager or RecordReplayManager(
            record_mode=record_mode,
            generate_log_files=generate_log_files,
        )
        self.logger = logger or CustomLogger.get_logger(
            __name__,
            generate_log_files=generate_log_files,
        )

    @conditional_retry
    def call_model(self, prompt: dict, stream=True):
        caller_name = get_original_caller()

        if "system" not in prompt or "user" not in prompt:
            raise KeyError("The prompt dictionary must contain 'system' and 'user' keys.")

        if prompt["system"] == "":
            messages = [{"role": "user", "content": prompt["user"]}]
        elif self.model in ["o1-preview", "o1-mini"]:
            messages = [
                {
                    "role": "user",
                    "content": prompt["system"] + "\n" + prompt["user"],
                }
            ]
        else:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]

        parameters = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }

        if self.model in ["o1-preview", "o1-mini", "o1", "o3-mini"]:
            stream = False
            parameters["temperature"] = 1
            parameters["stream"] = False
            parameters["max_completion_tokens"] = 2 * self.max_tokens
            parameters.pop("max_tokens", None)

        if (
            "ollama" in self.model
            or "huggingface" in self.model
            or self.model.startswith("openai/")
        ):
            parameters["api_base"] = self.api_base

        try:
            self.logger.info(f"📣 Calling LLM from {caller_name}()...")
            response = litellm.completion(**parameters)
        except Exception as error:
            self.logger.error(f"Error calling LLM model: {error}")
            raise error

        if stream:
            chunks = []
            self.logger.info("Streaming results from LLM model...")
            try:
                for chunk in response:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)
                    chunks.append(chunk)
                    time.sleep(0.01)
            except Exception as error:
                self.logger.error(f"Error calling LLM model during streaming: {error}")
                if self.enable_retry:
                    raise error

            assembled_response = litellm.stream_chunk_builder(chunks, messages=messages)
            print("\n")
            content = assembled_response["choices"][0]["message"]["content"]
            usage = assembled_response["usage"]
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
        else:
            content = response.choices[0].message.content
            self.logger.info("Printing results from LLM model...")
            print(content)
            usage = response.usage
            prompt_tokens = int(usage.prompt_tokens)
            completion_tokens = int(usage.completion_tokens)

        if "WANDB_API_KEY" in os.environ:
            try:
                trace = Trace(
                    name="inference_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    kind="llm",
                    inputs={
                        "user_prompt": prompt["user"],
                        "system_prompt": prompt["system"],
                    },
                    outputs={"model_response": content},
                )
                trace.log(name="inference")
            except Exception as error:
                self.logger.error(f"Error logging to W&B: {error}")

        if self.record_mode and self.source_file and self.test_file:
            self.record_replay_manager.record_response(
                self.source_file,
                self.test_file,
                prompt,
                content,
                prompt_tokens,
                completion_tokens,
                caller_name,
            )

        return content, prompt_tokens, completion_tokens