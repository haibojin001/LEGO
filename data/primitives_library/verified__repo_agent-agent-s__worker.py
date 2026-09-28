from functools import partial
import logging
import textwrap
from typing import Dict, List, Tuple

from gui_agents.s3.agents.grounding import ACI
from gui_agents.s3.core.module import BaseModule
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import (
    call_llm_safe,
    call_llm_formatted,
    parse_code_from_string,
    split_thinking_response,
    create_pyautogui_code,
)
from gui_agents.s3.utils.formatters import (
    SINGLE_ACTION_FORMATTER,
    CODE_VALID_FORMATTER,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        super().__init__(worker_engine_params, platform)

        self.temperature = worker_engine_params.get("temperature", 0.0)
        self.use_thinking = worker_engine_params.get("model", "") in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
        ]
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection

        self.reset()

    def reset(self):
        skipped_actions = ["set_cell_values"] if self.platform != "linux" else []

        environment = getattr(self.grounding_agent, "env", None)
        if not environment or not getattr(environment, "controller", None):
            skipped_actions.append("call_code_agent")

        system_prompt = (
            PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
                type(self.grounding_agent),
                skipped_actions=skipped_actions,
            ).replace("CURRENT_OS", self.platform)
        )

        self.generator_agent = self._create_agent(system_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []

    def flush_messages(self):
        engine_type = self.engine_params.get("engine_type", "")

        if engine_type in ["anthropic", "openai", "gemini"]:
            for agent in [self.generator_agent, self.reflection_agent]:
                if agent is None:
                    continue

                image_count = 0
                for message_index in range(len(agent.messages) - 1, -1, -1):
                    content = agent.messages[message_index].get("content", [])
                    for content_index in range(len(content) - 1, -1, -1):
                        item = content[content_index]
                        if "image" in item.get("type", ""):
                            image_count += 1
                            if image_count > self.max_trajectory_length:
                                del content[content_index]
        else:
            if (
                self.generator_agent is not None
                and len(self.generator_agent.messages)
                > 2 * self.max_trajectory_length + 1
            ):
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)

            if (
                self.reflection_agent is not None
                and len(self.reflection_agent.messages)
                > self.max_trajectory_length + 1
            ):
                self.reflection_agent.messages.pop(1)

    def _generate_reflection(self, instruction: str, obs: Dict) -> Tuple[str, str]:
        reflection = None
        reflection_thoughts = None

        if not self.enable_reflection:
            return reflection, reflection_thoughts

        if self.turn_count == 0:
            task_context = textwrap.dedent(
                f"""
                Task Description: {instruction}
                Current Trajectory below:
                """
            )
            self.reflection_agent.add_system_prompt(
                self.reflection_agent.system_prompt + "\n" + task_context
            )
            self.reflection_agent.add_message(
                text_content=(
                    "The initial screen is provided. No action has been taken yet."
                ),
                image_content=obs["screenshot"],
                role="user",
            )
            return reflection, reflection_thoughts

        self.reflection_agent.add_message(
            text_content=self.worker_history[-1],
            image_content=obs["screenshot"],
            role="user",
        )
        result = call_llm_safe(
            self.reflection_agent,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )

        if isinstance(result, tuple):
            result = result[0]

        reflection, reflection_thoughts = split_thinking_response(result)
        self.reflections.append(reflection)

        logger.info("REFLECTION THOUGHTS: %s", reflection_thoughts)
        logger.info("REFLECTION: %s", reflection)

        return reflection, reflection_thoughts

    @staticmethod
    def _extract_response_and_cost(result):
        if isinstance(result, tuple):
            response = result[0]
            cost = result[1] if len(result) > 1 else 0
            return response, cost
        return result, 0

    @staticmethod
    def _response_to_text(response):
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("response", "content", "text", "action", "code"):
                if key in response:
                    value = response[key]
                    if isinstance(value, str):
                        return value
            return str(response)
        return str(response)

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        self.cost_this_turn = 0

        self.grounding_agent.assign_screenshot(obs)
        self.grounding_agent.set_task_instruction(instruction)

        generator_message = (
            ""
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )

        if self.turn_count == 0:
            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION",
                    instruction,
                )
            )

        reflection, reflection_thoughts = self._generate_reflection(instruction, obs)

        if reflection:
            generator_message += (
                "REFLECTION: You may use this reflection on the previous action "
                "and overall trajectory:\n"
                f"{reflection}\n"
            )

        generator_message += (
            f"\nCurrent Text Buffer = "
            f"[{','.join(getattr(self.grounding_agent, 'notes', []))}]\n"
        )

        code_result = getattr(
            self.grounding_agent,
            "last_code_agent_result",
            None,
        )
        if code_result is not None:
            generator_message += "\nCODE AGENT RESULT:\n"
            generator_message += (
                f"Task/Subtask Instruction: "
                f"{code_result.get('task_instruction', '')}\n"
            )
            generator_message += (
                f"Steps Completed: {code_result.get('steps_executed', '')}\n"
            )
            generator_message += f"Max Steps: {code_result.get('budget', '')}\n"
            generator_message += (
                f"Completion Status: {code_result.get('status', '')}\n"
            )

            result_summary = code_result.get(
                "result",
                code_result.get("summary", code_result.get("message", "")),
            )
            if result_summary:
                generator_message += f"Result Summary: {result_summary}\n"

            self.grounding_agent.last_code_agent_result = None

        self.generator_agent.add_message(
            text_content=generator_message,
            image_content=obs["screenshot"],
            role="user",
        )

        llm_result = call_llm_formatted(
            self.generator_agent,
            formatter=SINGLE_ACTION_FORMATTER,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )
        raw_response, cost = self._extract_response_and_cost(llm_result)

        try:
            self.cost_this_turn += float(cost or 0)
        except (TypeError, ValueError):
            pass

        response_text = self._response_to_text(raw_response)
        response, thoughts = split_thinking_response(response_text)

        try:
            action_code = parse_code_from_string(response)
        except Exception:
            logger.exception("Unable to parse action code from worker response.")
            action_code = response

        if isinstance(action_code, (list, tuple)):
            action_code = "\n".join(str(item) for item in action_code)

        try:
            pyautogui_code = create_pyautogui_code(action_code)
        except Exception:
            logger.exception("Unable to create pyautogui code from worker action.")
            pyautogui_code = action_code

        history_entry = (
            f"Worker response:\n{response_text}\n\n"
            f"Executed action:\n{action_code}"
        )
        self.worker_history.append(history_entry)

        action = {
            "script": pyautogui_code,
            "action": action_code,
            "thoughts": thoughts,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
            "raw_response": response_text,
        }

        self.turn_count += 1
        self.flush_messages()

        logger.info("WORKER THOUGHTS: %s", thoughts)
        logger.info("WORKER ACTION: %s", action_code)

        return action, [action_code]