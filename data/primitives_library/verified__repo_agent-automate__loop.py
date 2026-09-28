from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..providers import ChatMessage, ProviderManager, ToolSpec
from ..providers.base import ToolCall
from ..store import Database
from ..tools import ToolRegistry
from .prompts import SYSTEM_PROMPT


@dataclass
class RunEvent:
    kind: str
    payload: dict = field(default_factory=dict)


@dataclass
class RunResult:
    id: str
    final: str
    events: list[RunEvent]


EventSink = Callable[[RunEvent], None]


class AgentLoop:
    def __init__(
        self,
        *,
        db: Database,
        providers: ProviderManager,
        registry: ToolRegistry,
        max_steps: int = 8,
    ):
        self.db = db
        self.providers = providers
        self.registry = registry
        self.max_steps = max_steps

    def _tool_specs(self, allowed: Iterable[str] | None = None) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for tool in self.registry.all():
            if allowed is not None and tool.name not in allowed:
                continue
            specs.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
        return specs

    def run(
        self,
        prompt: str,
        *,
        source: str = "web",
        model: str | None = None,
        on_event: EventSink | None = None,
        allowed_tools: Iterable[str] | None = None,
    ) -> RunResult:
        run_id = uuid.uuid4().hex
        self.db.create_run(id=run_id, source=source, prompt=prompt)
        events: list[RunEvent] = []

        def emit(event: RunEvent) -> None:
            events.append(event)
            self.db.append_trace(
                run_id,
                {"kind": event.kind, "payload": event.payload},
            )
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    pass

        client = self.providers.client()
        selected_model = model or self.providers.active_model() or ""
        tool_specs = self._tool_specs(allowed_tools)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        final_text = ""

        try:
            for step in range(self.max_steps):
                emit(RunEvent("thinking", {"step": step}))
                reply = client.chat(
                    messages,
                    model=selected_model,
                    tools=tool_specs,
                )

                if reply.content:
                    emit(RunEvent("message", {"text": reply.content}))

                if not reply.tool_calls:
                    final_text = reply.content or ""
                    emit(RunEvent("final", {"text": final_text}))
                    break

                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=reply.content,
                        tool_calls=reply.tool_calls,
                    )
                )

                for tool_call in reply.tool_calls:
                    self._dispatch(tool_call, messages, emit)
            else:
                final_text = (
                    "Stopped: reached max tool-use steps without a final answer."
                )
                emit(RunEvent("final", {"text": final_text, "truncated": True}))

            self.db.finish_run(run_id, status="done", result=final_text)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            emit(RunEvent("error", {"message": error}))
            self.db.finish_run(run_id, status="error", result=error)
            raise

        return RunResult(id=run_id, final=final_text, events=events)

    def _dispatch(
        self,
        tc: ToolCall,
        messages: list[ChatMessage],
        emit: EventSink,
    ) -> None:
        emit(
            RunEvent(
                "tool_call",
                {"id": tc.id, "name": tc.name, "args": tc.arguments},
            )
        )

        tool = self.registry.get(tc.name)
        if not tool:
            result_text = json.dumps({"error": f"unknown tool: {tc.name}"})
        elif tool.tier == "pro" and not self._has_pro_session():
            result_text = json.dumps(
                {
                    "error": "needs_pro_subscription",
                    "message": (
                        f"The tool '{tool.name}' requires a Pro subscription. "
                        "Tell the user they can sign in via Settings → autoMate Cloud."
                    ),
                }
            )
        else:
            try:
                result = tool.call(tc.arguments)
                result_text = (
                    result
                    if isinstance(result, str)
                    else json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )[:8000]
                )
            except Exception as exc:
                result_text = json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}
                )

        emit(
            RunEvent(
                "tool_result",
                {
                    "id": tc.id,
                    "name": tc.name,
                    "result": result_text,
                },
            )
        )
        messages.append(
            ChatMessage(
                role="tool",
                tool_call_id=tc.id,
                name=tc.name,
                content=result_text,
            )
        )

    def _has_pro_session(self) -> bool:
        try:
            from .. import auth

            return auth.get_session(self.db) is not None
        except Exception:
            return False