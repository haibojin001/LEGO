import inspect
import json
import logging
import os
import platform
from typing import Dict, List, Optional, Tuple

from gui_agents.s2.agents.grounding import ACI
from gui_agents.s2.agents.worker import Worker
from gui_agents.s2.agents.manager import Manager
from gui_agents.s2.utils.common_utils import Node
from gui_agents.utils import download_kb_data
from gui_agents.s2.core.engine import (
    OpenAIEmbeddingEngine,
    GeminiEmbeddingEngine,
    AzureOpenAIEmbeddingEngine,
)

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        action_space: str = "pyautogui",
        observation_type: str = "a11y_tree",
        search_engine: str = "perplexica",
    ):
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.engine = search_engine

    def reset(self) -> None:
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        pass

    def update_narrative_memory(self, trajectory: str) -> None:
        pass

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        pass


class AgentS2(UIAgent):
    """Agent that uses hierarchical planning and DAG modeling for UI automation."""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        action_space: str = "pyautogui",
        observation_type: str = "mixed",
        search_engine: Optional[str] = None,
        memory_root_path: str = os.getcwd(),
        use_default_kb: bool = False,
        memory_folder_name: str = "kb_s2",
        kb_release_tag: str = "v0.2.2",
        embedding_engine_type: str = "openai",
        embedding_engine_params: Dict = {},
    ):
        super().__init__(
            engine_params,
            grounding_agent,
            platform,
            action_space,
            observation_type,
            search_engine,
        )

        self.memory_root_path = memory_root_path
        self.memory_folder_name = memory_folder_name
        self.kb_release_tag = kb_release_tag
        self.local_kb_path = os.path.join(memory_root_path, memory_folder_name)

        if use_default_kb:
            platform_kb_path = os.path.join(self.local_kb_path, self.platform)
            if not os.path.exists(platform_kb_path):
                print("Downloading Agent S2's default knowledge base...")
                download_kb_data(
                    version="s2",
                    release_tag=kb_release_tag,
                    download_dir=self.local_kb_path,
                    platform=self.platform,
                )
                print(
                    f"Successfully completed download of knowledge base for version s2, "
                    f"tag {self.kb_release_tag}, platform {self.platform}."
                )
            else:
                print(
                    f"Path local_kb_path {self.local_kb_path} already exists. "
                    "Skipping download."
                )
                print(
                    "If you'd like to re-download the initial knowledge base, please "
                    f"delete the existing knowledge base at {self.local_kb_path}."
                )
                print(
                    "Note, the knowledge is continually updated during inference. "
                    "Deleting the knowledge base will wipe out all experience gained "
                    "since the last knowledge base download."
                )

        if embedding_engine_type == "openai":
            self.embedding_engine = OpenAIEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "gemini":
            self.embedding_engine = GeminiEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "azure":
            self.embedding_engine = AzureOpenAIEmbeddingEngine(
                **embedding_engine_params
            )

        self.reset()

    def reset(self) -> None:
        self.planner = Manager(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path=self.local_kb_path,
            embedding_engine=self.embedding_engine,
            search_engine=self.engine,
            platform=self.platform,
        )
        self.executor = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path=self.local_kb_path,
            embedding_engine=self.embedding_engine,
            platform=self.platform,
        )

        self.requires_replan: bool = True
        self.needs_next_subtask: bool = True
        self.step_count: int = 0
        self.turn_count: int = 0
        self.failure_subtask: Optional[Node] = None
        self.should_send_action: bool = False
        self.completed_tasks: List[Node] = []
        self.current_subtask: Optional[Node] = None
        self.subtasks: List[Node] = []
        self.search_query: str = ""
        self.subtask_status: str = "Start"

    def reset_executor_state(self) -> None:
        self.executor.reset()
        self.step_count = 0

    @staticmethod
    def _call_with_supported_arguments(method, values: Dict):
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(**values)

        parameters = signature.parameters
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return method(**values)

        aliases = {
            "task": "instruction",
            "query": "search_query",
            "current_subtask": "subtask",
            "task_node": "subtask",
            "failed_task": "failed_subtask",
            "completed_tasks": "completed_subtasks_list",
            "completed_subtasks": "completed_subtasks_list",
            "status": "subtask_status",
            "state": "observation",
            "obs": "observation",
        }

        kwargs = {}
        for name, parameter in parameters.items():
            if name == "self":
                continue
            key = name if name in values else aliases.get(name)
            if key in values:
                kwargs[name] = values[key]
            elif parameter.default is inspect.Parameter.empty:
                raise TypeError(f"Unable to supply required argument: {name}")

        return method(**kwargs)

    @staticmethod
    def _unpack_executor_result(result):
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0] or {}, result[1] or [], result[2] if len(result) > 2 else None
            if len(result) == 1:
                return result[0] or {}, [], None
        if isinstance(result, dict):
            actions = result.get("actions", result.get("action", []))
            return result, actions or [], result.get("subtask_status")
        return {}, result or [], None

    @staticmethod
    def _unpack_evaluator_result(result):
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0] or {}, result[1]
            if len(result) == 1:
                return result[0] or {}, None
        if isinstance(result, dict):
            return result, result.get("subtask_status", result.get("status"))
        if isinstance(result, bool):
            return {}, "Done" if result else "Failed"
        if result is not None:
            return {}, result
        return {}, None

    @staticmethod
    def _is_done_action(actions) -> bool:
        if isinstance(actions, str):
            return actions.strip().upper() == "DONE"
        return (
            isinstance(actions, (list, tuple))
            and len(actions) == 1
            and isinstance(actions[0], str)
            and actions[0].strip().upper() == "DONE"
        )

    @staticmethod
    def _is_failure_action(actions) -> bool:
        if isinstance(actions, str):
            return actions.strip().upper() in {"FAIL", "FAILED"}
        return (
            isinstance(actions, (list, tuple))
            and len(actions) == 1
            and isinstance(actions[0], str)
            and actions[0].strip().upper() in {"FAIL", "FAILED"}
        )

    @staticmethod
    def _status_is_failure(status) -> bool:
        if status is None:
            return False
        value = str(status).lower()
        return any(token in value for token in ("fail", "error", "reject"))

    @staticmethod
    def _status_is_complete(status) -> bool:
        if status is None:
            return True
        value = str(status).lower()
        return any(
            token in value
            for token in ("done", "pass", "success", "complete", "finish")
        )

    def _evaluate_subtask(self, instruction: str, observation: Dict):
        values = {
            "instruction": instruction,
            "observation": observation,
            "subtask": self.current_subtask,
            "search_query": self.search_query,
            "subtask_status": self.subtask_status,
            "step_count": self.step_count,
        }

        for component in (self.planner, self.executor):
            for name in (
                "evaluate_subtask",
                "evaluate",
                "check_subtask_completion",
                "verify_subtask",
            ):
                method = getattr(component, name, None)
                if callable(method):
                    return self._unpack_evaluator_result(
                        self._call_with_supported_arguments(method, values)
                    )

        return {}, "Done"

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        planner_info = {}
        executor_info = {}
        evaluator_info = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }
        actions: List[str] = []

        self.turn_count += 1
        self.should_send_action = False

        while not self.should_send_action:
            self.subtask_status = "In"

            if self.requires_replan:
                logger.info("(RE)PLANNING...")
                planner_info, self.subtasks = self.planner.get_action_queue(
                    instruction=instruction,
                    observation=observation,
                    failed_subtask=self.failure_subtask,
                    completed_subtasks_list=self.completed_tasks,
                )
                self.subtasks = list(self.subtasks or [])
                self.requires_replan = False
                self.failure_subtask = None
                self.needs_next_subtask = True

                if isinstance(planner_info, dict):
                    self.search_query = planner_info.get(
                        "search_query",
                        planner_info.get("query", self.search_query),
                    )

            if self.needs_next_subtask:
                if not self.subtasks:
                    actions = ["DONE"]
                    self.subtask_status = "Done"
                    self.should_send_action = True
                    break

                self.current_subtask = self.subtasks.pop(0)
                self.needs_next_subtask = False
                self.reset_executor_state()

            result = self._call_with_supported_arguments(
                self.executor.generate_next_action,
                {
                    "instruction": instruction,
                    "search_query": self.search_query,
                    "subtask": self.current_subtask,
                    "observation": observation,
                    "subtask_status": self.subtask_status,
                    "step_count": self.step_count,
                },
            )
            executor_info, actions, returned_status = self._unpack_executor_result(result)
            self.step_count += 1

            if returned_status is not None:
                self.subtask_status = str(returned_status)

            if self._is_failure_action(actions) or self._status_is_failure(
                returned_status
            ):
                self.failure_subtask = self.current_subtask
                self.requires_replan = True
                self.needs_next_subtask = True
                self.should_send_action = False
                continue

            if self._is_done_action(actions):
                evaluator_info, evaluator_status = self._evaluate_subtask(
                    instruction, observation
                )
                if self._status_is_failure(evaluator_status):
                    self.failure_subtask = self.current_subtask
                elif self._status_is_complete(evaluator_status):
                    if self.current_subtask is not None:
                        self.completed_tasks.append(self.current_subtask)
                else:
                    self.failure_subtask = self.current_subtask

                self.requires_replan = True
                self.needs_next_subtask = True
                self.should_send_action = False
                continue

            self.should_send_action = True

        info = {}
        info.update(planner_info or {})
        info.update(executor_info or {})
        info.update(evaluator_info or {})
        return info, actions

    def update_narrative_memory(self, trajectory: str) -> None:
        for component in (self.planner, self.executor):
            method = getattr(component, "update_narrative_memory", None)
            if callable(method):
                self._call_with_supported_arguments(
                    method, {"trajectory": trajectory}
                )
                return

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        for component in (self.planner, self.executor):
            method = getattr(component, "update_episodic_memory", None)
            if callable(method):
                result = self._call_with_supported_arguments(
                    method,
                    {
                        "meta_data": meta_data,
                        "subtask_trajectory": subtask_trajectory,
                    },
                )
                return (
                    subtask_trajectory
                    if result is None
                    else result
                )
        return subtask_trajectory