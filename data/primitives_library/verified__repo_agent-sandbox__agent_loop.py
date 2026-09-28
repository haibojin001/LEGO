import inspect
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession
from openai import AzureOpenAI, OpenAI


DEFAULT_SYSTEM_PROMPT = """You are an AI assistant with access to tools.

When given a task, you MUST:
1. Use the available tools to complete the task
2. Provide summary of each step in your approach, wrapped in <summary> tags
3. Provide feedback on the tools provided, wrapped in <feedback> tags
4. Provide your final response, wrapped in <response> tags
5. Don't use VLM tools, like screenshot, etc.

Summary Requirements:
- In your <summary> tags, you must explain:
  - The steps you took to complete the task
  - Which tools you used, in what order, and why
  - The inputs you provided to each tool
  - The outputs you received from each tool
  - A summary for how you arrived at the response

Feedback Requirements:
- In your <feedback> tags, provide constructive feedback on the tools:
  - Comment on tool names: Are they clear and descriptive?
  - Comment on input parameters: Are they well-documented? Are required vs optional parameters clear?
  - Comment on descriptions: Do they accurately describe what the tool does?
  - Comment on any errors encountered during tool usage: Did the tool fail to execute? Did the tool return too many tokens?
  - Identify specific areas for improvement and explain WHY they would help
  - Be specific and actionable in your suggestions

Response Requirements:
- Your response should be concise and directly address what was asked
- Always wrap your final response in <response> tags
- If you cannot solve the task return <response>NOT_FOUND</response>
- For numeric responses, provide just the number
- For IDs, provide just the ID
- For names or text, provide the exact text requested
- Your response should go last"""


class BaseAgentLoop(ABC):
    """Abstract base class for agent loop implementations."""

    def __init__(
        self,
        mcp_session: ClientSession,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.mcp_session = mcp_session
        self.system_prompt = system_prompt

    @abstractmethod
    async def run(
        self,
        prompt: str,
        tools: List[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute an agent loop and return its response and tool metrics."""

    @staticmethod
    def _get_value(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        if not isinstance(text, str):
            return text
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _tool_result_to_text(result: Any) -> str:
        if result is None:
            return ""

        content = getattr(result, "content", result)
        if isinstance(content, str):
            return content

        if not isinstance(content, (list, tuple)):
            text = getattr(content, "text", None)
            if text is not None:
                return str(text)
            if isinstance(content, dict) and "text" in content:
                return str(content["text"])
            try:
                return json.dumps(content)
            except (TypeError, ValueError):
                return str(content)

        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    try:
                        parts.append(json.dumps(item))
                    except (TypeError, ValueError):
                        parts.append(str(item))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
                elif hasattr(item, "model_dump"):
                    try:
                        parts.append(json.dumps(item.model_dump()))
                    except (TypeError, ValueError):
                        parts.append(str(item))
                else:
                    parts.append(str(item))
        return "\n".join(parts)

    async def _execute_tool_call(
        self,
        tool_call: Any,
        tool_metrics: Dict[str, Any],
    ) -> Tuple[str, str]:
        function = self._get_value(tool_call, "function", {})
        tool_name = self._get_value(function, "name")
        arguments = self._get_value(function, "arguments", "{}")

        metric_name = tool_name or "unknown_tool"
        metric = tool_metrics.setdefault(
            metric_name,
            {"count": 0, "total_time": 0.0, "errors": 0},
        )

        if isinstance(arguments, dict):
            tool_arguments = arguments
        else:
            try:
                tool_arguments = json.loads(arguments or "{}")
            except (json.JSONDecodeError, TypeError) as exc:
                metric["count"] += 1
                metric["errors"] += 1
                return (
                    tool_name or metric_name,
                    f"Error parsing arguments for tool {tool_name}: {exc}",
                )

        started_at = time.time()
        successful = True
        try:
            result = self.mcp_session.call_tool(tool_name, tool_arguments)
            if inspect.isawaitable(result):
                result = await result
            output = self._tool_result_to_text(result)
            successful = not bool(
                self._get_value(
                    result,
                    "isError",
                    self._get_value(result, "is_error", False),
                )
            )
            if not successful and not output:
                output = f"Tool {tool_name} returned an error"
        except Exception as exc:
            output = f"Error executing tool {tool_name}: {exc}"
            successful = False

        metric["count"] += 1
        metric["total_time"] += time.time() - started_at
        if not successful:
            metric["errors"] += 1

        return tool_name or metric_name, output

    @staticmethod
    def _message_to_dict(message: Any, tool_calls: Any) -> Dict[str, Any]:
        result = {
            "role": BaseAgentLoop._get_value(message, "role", "assistant"),
            "content": BaseAgentLoop._get_value(message, "content"),
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    async def _run_loop(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        client: Any,
        model: str,
        max_iterations: int,
        strip_thinking_tags: bool = False,
        temperature: Any = None,
    ) -> Tuple[str, Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tool_metrics: Dict[str, Any] = {}
        last_assistant_content = ""

        for _ in range(max_iterations):
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": 4096,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            if inspect.isawaitable(response):
                response = await response

            choices = self._get_value(response, "choices", [])
            if not choices:
                return "", tool_metrics

            message = self._get_value(choices[0], "message")
            if message is None:
                return "", tool_metrics

            content = self._get_value(message, "content", "")
            last_assistant_content = content or ""
            tool_calls = self._get_value(message, "tool_calls")
            messages.append(self._message_to_dict(message, tool_calls))

            if not tool_calls:
                final_content = last_assistant_content
                if strip_thinking_tags:
                    final_content = self._strip_thinking_tags(final_content)

                missing_tags = [
                    tag
                    for tag in ("<response>", "<summary>", "<feedback>")
                    if tag not in final_content
                ]
                if not missing_tags:
                    return final_content, tool_metrics

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "ERROR: Your response is missing required tags: "
                            + ", ".join(missing_tags)
                            + ". Please provide the complete response using "
                            "<summary>, <feedback>, and <response> tags."
                        ),
                    }
                )
                continue

            for tool_call in tool_calls:
                _, output = await self._execute_tool_call(tool_call, tool_metrics)
                tool_call_id = self._get_value(tool_call, "id")
                tool_message: Dict[str, Any] = {
                    "role": "tool",
                    "content": output,
                }
                if tool_call_id is not None:
                    tool_message["tool_call_id"] = tool_call_id
                messages.append(tool_message)

        if strip_thinking_tags:
            last_assistant_content = self._strip_thinking_tags(last_assistant_content)
        return last_assistant_content, tool_metrics


class AzureOpenAIAgentLoop(BaseAgentLoop):
    """Agent loop implementation using Azure OpenAI."""

    def __init__(
        self,
        mcp_session: ClientSession,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        azure_endpoint: str = None,
        azure_api_key: str = None,
        azure_deployment: str = None,
        azure_api_version: str = None,
        max_iterations: int = 50,
    ):
        super().__init__(mcp_session, system_prompt)
        self.azure_endpoint = azure_endpoint or os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com"
        )
        self.azure_api_key = azure_api_key or os.getenv(
            "AZURE_OPENAI_API_KEY", "your-api-key"
        )
        self.azure_deployment = azure_deployment or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT", "gpt-4"
        )
        self.azure_api_version = azure_api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        self.max_iterations = max_iterations
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
        )

    async def run(
        self,
        prompt: str,
        tools: List[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        return await self._run_loop(
            prompt=prompt,
            tools=tools,
            client=self.client,
            model=self.azure_deployment,
            max_iterations=self.max_iterations,
        )


class OpenAIAgentLoop(BaseAgentLoop):
    """Agent loop implementation using OpenAI or an OpenAI-compatible API."""

    def __init__(
        self,
        mcp_session: ClientSession,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_iterations: int = 50,
        temperature: Optional[float] = 0.0,
        openai_api_key: str = None,
        openai_base_url: str = None,
    ):
        super().__init__(mcp_session, system_prompt)

        api_key = api_key if api_key is not None else openai_api_key
        base_url = base_url if base_url is not None else openai_base_url

        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "your-api-key")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_iterations = max_iterations
        self.temperature = temperature

        # Retain these aliases for callers that use the provider-specific names.
        self.openai_api_key = self.api_key
        self.openai_base_url = self.base_url

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _effective_temperature(self) -> Optional[float]:
        """Return a temperature compatible with the selected API provider."""
        if self.temperature is None:
            return None

        if self.base_url and "minimax" in self.base_url.lower():
            try:
                if float(self.temperature) <= 0:
                    return 0.01
            except (TypeError, ValueError):
                pass
        return self.temperature

    async def run(
        self,
        prompt: str,
        tools: List[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        return await self._run_loop(
            prompt=prompt,
            tools=tools,
            client=self.client,
            model=self.model,
            max_iterations=self.max_iterations,
            strip_thinking_tags=True,
            temperature=self._effective_temperature(),
        )


class LangGraphAgentLoop(BaseAgentLoop):
    """Placeholder for a future LangGraph-based agent-loop implementation."""

    async def run(
        self,
        prompt: str,
        tools: List[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError(
            "LangGraphAgentLoop is not implemented in this version of the library."
        )