import logging
import os
import platform
import re
from typing import Dict, List, Tuple

from gui_agents.s1.aci.ACI import ACI
from gui_agents.s1.core.BaseModule import BaseModule
from gui_agents.s1.core.Knowledge import KnowledgeBase
from gui_agents.s1.core.ProceduralMemory import PROCEDURAL_MEMORY
from gui_agents.s1.utils import common_utils
from gui_agents.s1.utils.common_utils import Node, calculate_tokens, call_llm_safe

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        local_kb_path: str,
        platform: str = platform.system().lower(),
        search_engine: str = "perplexica",
        enable_reflection: bool = True,
        use_subtask_experience: bool = True,
    ):
        super().__init__(engine_params, platform)
        self.grounding_agent = grounding_agent
        self.local_kb_path = local_kb_path
        self.enable_reflection = enable_reflection
        self.search_engine = search_engine
        self.use_subtask_experience = use_subtask_experience
        self.reset()

    def flush_messages(self, n):
        for agent in (self.generator_agent,):
            if len(agent.messages) > 2 * n + 1:
                agent.remove_message_at(1)
                agent.remove_message_at(1)

    def reset(self):
        worker_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
            type(self.grounding_agent)
        ).replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(worker_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )
        self.knowledge_base = KnowledgeBase(
            local_kb_path=self.local_kb_path,
            platform=self.platform,
            engine_params=self.engine_params,
        )

        self.turn_count = 0
        self.planner_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.tree_inputs = []
        self.screenshot_inputs = []

    def remove_ids_from_history(self):
        for message in self.generator_agent.messages:
            if message["role"] != "user":
                continue

            for content in message["content"]:
                if content["type"] != "text":
                    continue

                lines = content["text"].splitlines()
                lines = [re.sub(r"^\d+\s+", "", line) for line in lines]
                content["text"] = "\n".join(lines).replace("id\t", "")

    def generate_next_action(
        self,
        instruction: str,
        search_query: str,
        subtask: str,
        subtask_info: str,
        future_tasks: List[Node],
        done_task: List[Node],
        obs: Dict,
    ) -> Tuple[Dict, List]:
        agent = self.grounding_agent
        self.active_apps = agent.get_active_apps(obs)

        if self.turn_count == 0:
            if self.use_subtask_experience:
                query = (
                    "Task:\n"
                    + search_query
                    + "\n\nSubtask: "
                    + subtask
                    + "\nSubtask Instruction: "
                    + subtask_info
                )
                similar_subtask, experience = (
                    self.knowledge_base.retrieve_episodic_experience(query)
                )
                logger.info(
                    "SIMILAR SUBTASK EXPERIENCE: %s",
                    similar_subtask + "\n" + experience.strip(),
                )
                instruction += (
                    "\nYou may refer to some similar subtask experience if you think "
                    "they are useful. {}".format(similar_subtask + "\n" + experience)
                )

            prompt = self.generator_agent.system_prompt
            prompt = prompt.replace("SUBTASK_DESCRIPTION", subtask)
            prompt = prompt.replace("TASK_DESCRIPTION", instruction)
            prompt = prompt.replace(
                "FUTURE_TASKS", ", ".join(task.name for task in future_tasks)
            )
            prompt = prompt.replace(
                "DONE_TASKS", ",".join(task.name for task in done_task)
            )
            self.generator_agent.add_system_prompt(prompt)

        reflection = None
        if self.enable_reflection and self.turn_count > 0:
            reflection_prompt = (
                "Task Description: "
                + subtask
                + " Instruction: "
                + subtask_info
                + "\nCurrent Trajectory: "
                + "\n\n".join(self.planner_history)
                + "\n"
            )
            self.reflection_agent.add_message(reflection_prompt)
            reflection = call_llm_safe(self.reflection_agent)
            self.reflections.append(reflection)
            self.reflection_agent.add_message(reflection)
            logger.info("REFLECTION: %s", reflection)

        tree_input = agent.linearize_and_annotate_tree(obs)
        self.tree_inputs.append(tree_input)
        self.screenshot_inputs.append(obs["screenshot"])

        self.remove_ids_from_history()

        message = ""
        if reflection:
            message += (
                "\nYou may use the reflection on the previous trajectory: "
                + reflection
                + "\n"
            )

        message += (
            f"Accessibility Tree: {tree_input}\n"
            f"Text Buffer = [{','.join(agent.notes)}]. "
            f"The current open applications are {agent.get_active_apps(obs)} and "
            f"the active app is {agent.get_top_app(obs)}.\n"
        )

        print("ACTIVE APP IS: ", agent.get_top_app(obs))

        if self.turn_count == 0:
            message += f"Remeber only complete the subtask: {subtask}\n"
            message += (
                "You can use this extra information for completing the current "
                f"subtask: {subtask_info}.\n"
            )

        logger.info("GENERATOR MESSAGE: %s", message)
        self.generator_agent.add_message(message, image_content=obs["screenshot"])

        plan = call_llm_safe(self.generator_agent)
        self.planner_history.append(plan)
        logger.info("PLAN: %s", plan)
        self.generator_agent.add_message(plan)

        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)
        self.cost_this_turn += (
            input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)
        )
        logger.info("EXECTUOR COST: %s", self.cost_this_turn)

        code = common_utils.parse_single_code_from_string(
            plan.split("Grounded Action")[-1]
        )
        code = common_utils.sanitize_code(code)
        code = common_utils.extract_first_agent_function(code)
        action = eval(code)

        self.turn_count += 1
        return action, [plan]