import json
import logging
import os
import platform
from typing import Dict, List, Optional, Tuple

from gui_agents.s1.aci.ACI import ACI
from gui_agents.s1.core.Manager import Manager
from gui_agents.s1.core.Worker import Worker
from gui_agents.s1.utils.common_utils import Node
from gui_agents.utils import download_kb_data

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents."""

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


class GraphSearchAgent(UIAgent):
    """Hierarchical graph-search UI automation agent."""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        action_space: str = "pyatuogui",
        observation_type: str = "mixed",
        search_engine: Optional[str] = None,
        memory_root_path: str = os.getcwd(),
        memory_folder_name: str = "kb_s1",
        kb_release_tag: str = "v0.2.2",
    ):
        super().__init__(
            engine_params=engine_params,
            grounding_agent=grounding_agent,
            platform=platform,
            action_space=action_space,
            observation_type=observation_type,
            search_engine=search_engine,
        )

        self.memory_root_path = memory_root_path
        self.memory_folder_name = memory_folder_name
        self.kb_release_tag = kb_release_tag
        self.local_kb_path = os.path.join(memory_root_path, memory_folder_name)

        print("Downloading knowledge base initial Agent-S knowledge...")
        if not os.path.exists(self.local_kb_path):
            download_kb_data(
                version="s1",
                release_tag=self.kb_release_tag,
                download_dir=self.local_kb_path,
                platform=self.platform,
            )
            print(
                "Successfully completed download of knowledge base for "
                f"version s1, tag {self.kb_release_tag}, platform {self.platform}."
            )
        else:
            print(
                f"Path local_kb_path {self.local_kb_path} already exists. "
                "Skipping download."
            )
            print(
                "If you'd like to re-download the initial knowledge base, "
                f"please delete the existing knowledge base at {self.local_kb_path}."
            )
            print(
                "Note, the knowledge is continually updated during inference. "
                "Deleting the knowledge base will wipe out all experience gained "
                "since the last knowledge base download."
            )

        self.reset()

    def reset(self) -> None:
        self.planner = Manager(
            self.engine_params,
            self.grounding_agent,
            platform=self.platform,
            search_engine=self.engine,
            local_kb_path=self.local_kb_path,
        )
        self.executor = Worker(
            self.engine_params,
            self.grounding_agent,
            platform=self.platform,
            local_kb_path=self.local_kb_path,
        )

        self.requires_replan = True
        self.needs_next_subtask = True
        self.step_count = 0
        self.turn_count = 0
        self.failure_feedback = ""
        self.should_send_action = False
        self.completed_tasks: List[Node] = []
        self.current_subtask: Optional[Node] = None
        self.subtasks: List[Node] = []
        self.search_query = ""
        self.subtask_status = "Start"

    def reset_executor_state(self) -> None:
        self.executor.reset()
        self.step_count = 0

    def _evaluate_subtask(
        self,
        instruction: str,
        observation: Dict,
    ) -> Tuple[Dict, str]:
        default_info = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }

        evaluator = getattr(self.planner, "evaluate_subtask", None)
        if not callable(evaluator):
            return default_info, "Pass"

        try:
            result = evaluator(
                instruction=instruction,
                subtask=self.current_subtask,
                observation=observation,
            )
        except TypeError:
            try:
                result = evaluator(instruction, self.current_subtask, observation)
            except TypeError:
                result = evaluator(self.current_subtask, observation)

        if isinstance(result, tuple) and len(result) == 2:
            info, status = result
            if isinstance(info, dict):
                default_info.update(info)
            return default_info, status

        if isinstance(result, dict):
            default_info.update(result)
            return default_info, str(
                result.get("subtask_status", result.get("status", "Pass"))
            )

        if isinstance(result, str):
            return default_info, result

        return default_info, "Pass"

    @staticmethod
    def _is_success_status(status: str) -> bool:
        return str(status).strip().lower() in {
            "pass",
            "passed",
            "success",
            "succeeded",
            "done",
            "complete",
            "completed",
        }

    @staticmethod
    def _is_failure_status(status: str) -> bool:
        return str(status).strip().lower() in {
            "fail",
            "failed",
            "failure",
            "error",
            "unsuccessful",
        }

    def _make_info(
        self,
        planner_info: Dict,
        executor_info: Dict,
        evaluator_info: Dict,
    ) -> Dict:
        return {
            "planner_info": planner_info,
            "executor_info": executor_info,
            "evaluator_info": evaluator_info,
            "current_subtask": self.current_subtask,
            "completed_tasks": self.completed_tasks,
            "subtasks": self.subtasks,
            "search_query": self.search_query,
            "subtask_status": self.subtask_status,
            "step_count": self.step_count,
            "turn_count": self.turn_count,
        }

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        planner_info: Dict = {}
        executor_info: Dict = {}
        evaluator_info: Dict = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }
        actions: List[str] = []

        self.should_send_action = False

        while not self.should_send_action:
            self.subtask_status = "In"

            if self.requires_replan:
                logger.info("(RE)PLANNING...")
                planner_info, self.subtasks = self.planner.get_action_queue(
                    instruction=instruction,
                    observation=observation,
                    failure_feedback=self.failure_feedback,
                )
                self.subtasks = list(self.subtasks or [])
                self.requires_replan = False
                self.failure_feedback = ""
                self.search_query = planner_info.get("search_query", "")

            if self.needs_next_subtask:
                if not self.subtasks:
                    self.subtask_status = "Done"
                    return (
                        self._make_info(
                            planner_info,
                            executor_info,
                            evaluator_info,
                        ),
                        [],
                    )

                logger.info("GETTING NEXT SUBTASK...")
                self.current_subtask = self.subtasks.pop(0)
                logger.info("NEXT SUBTASK: %s", self.current_subtask)
                self.needs_next_subtask = False
                self.subtask_status = "Start"

            executor_info, actions = self.executor.generate_next_action(
                instruction=instruction,
                search_query=self.search_query,
                subtask=self.current_subtask,
                completed_tasks=self.completed_tasks,
                remaining_subtasks=self.subtasks,
                observation=observation,
            )
            executor_info = executor_info or {}
            actions = list(actions or [])
            self.step_count += 1
            self.turn_count += 1

            action_text = "\n".join(str(action) for action in actions).strip().lower()
            executor_status = str(
                executor_info.get(
                    "subtask_status",
                    executor_info.get("status", executor_info.get("executor_status", "")),
                )
            )

            failed = (
                self._is_failure_status(executor_status)
                or action_text in {"fail", "failure"}
                or action_text.startswith("fail\n")
            )
            done = (
                self._is_success_status(executor_status)
                or action_text in {"done", "completed", "complete"}
                or action_text.startswith("done\n")
                or not actions
            )

            if failed:
                self.failure_feedback = str(
                    executor_info.get(
                        "failure_feedback",
                        executor_info.get(
                            "feedback",
                            executor_info.get("executor_response", ""),
                        ),
                    )
                )
                self.requires_replan = True
                self.needs_next_subtask = True
                self.subtask_status = "Fail"
                self.reset_executor_state()
                continue

            if done:
                evaluator_info, self.subtask_status = self._evaluate_subtask(
                    instruction,
                    observation,
                )

                if self._is_failure_status(self.subtask_status):
                    self.failure_feedback = str(
                        evaluator_info.get(
                            "failure_feedback",
                            evaluator_info.get(
                                "feedback",
                                evaluator_info.get("obs_evaluator_response", ""),
                            ),
                        )
                    )
                    self.requires_replan = True
                    self.needs_next_subtask = True
                    self.reset_executor_state()
                    continue

                self.completed_tasks.append(self.current_subtask)
                self.needs_next_subtask = True
                self.subtask_status = "Done"
                self.reset_executor_state()
                continue

            self.should_send_action = True

        return (
            self._make_info(planner_info, executor_info, evaluator_info),
            actions,
        )

    def update_narrative_memory(self, trajectory: str) -> None:
        updater = getattr(self.planner, "update_narrative_memory", None)
        if callable(updater):
            updater(trajectory)
            return

        memory_path = os.path.join(self.local_kb_path, "narrative_memory.json")
        data = []
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, json.JSONDecodeError):
                data = []

        if not isinstance(data, list):
            data = [data]
        data.append(trajectory)

        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        updater = getattr(self.executor, "update_episodic_memory", None)
        if callable(updater):
            return updater(meta_data, subtask_trajectory)

        updater = getattr(self.planner, "update_episodic_memory", None)
        if callable(updater):
            return updater(meta_data, subtask_trajectory)

        memory_path = os.path.join(self.local_kb_path, "episodic_memory.json")
        memories = []
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as file:
                    memories = json.load(file)
            except (OSError, json.JSONDecodeError):
                memories = []

        if not isinstance(memories, list):
            memories = [memories]

        record = {
            "meta_data": meta_data,
            "subtask_trajectory": subtask_trajectory,
        }
        memories.append(record)

        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, "w", encoding="utf-8") as file:
            json.dump(memories, file, indent=2)

        return subtask_trajectory