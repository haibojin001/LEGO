import logging
import platform
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from gui_agents.s2.agents.grounding import ACI
from gui_agents.s2.core.engine import OpenAIEmbeddingEngine
from gui_agents.s2.core.knowledge import KnowledgeBase
from gui_agents.s2.core.module import BaseModule
from gui_agents.s2.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2.utils.common_utils import (
    Dag,
    Node,
    calculate_tokens,
    call_llm_safe,
    parse_dag,
)

logger = logging.getLogger("desktopenv.agent")

NUM_IMAGE_TOKEN = 1105


class Manager(BaseModule):
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        local_kb_path: str,
        embedding_engine,
        search_engine: Optional[str] = None,
        multi_round: bool = False,
        platform: str = platform.system().lower(),
    ):
        super().__init__(engine_params, platform)

        self.grounding_agent = grounding_agent

        self.generator_agent = self._create_agent(
            PROCEDURAL_MEMORY.COMBINED_MANAGER_PROMPT
        )
        self.dag_translator_agent = self._create_agent(
            PROCEDURAL_MEMORY.DAG_TRANSLATOR_PROMPT
        )
        self.narrative_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.TASK_SUMMARIZATION_PROMPT
        )
        self.episode_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.SUBTASK_SUMMARIZATION_PROMPT
        )

        self.local_kb_path = local_kb_path
        self.embedding_engine = embedding_engine
        self.knowledge_base = KnowledgeBase(
            embedding_engine=self.embedding_engine,
            local_kb_path=self.local_kb_path,
            platform=platform,
            engine_params=engine_params,
        )

        self.planner_history = []
        self.turn_count = 0
        self.search_engine = search_engine
        self.multi_round = multi_round

    def summarize_episode(self, trajectory):
        self.episode_summarization_agent.add_message(trajectory, role="user")
        summary = call_llm_safe(self.episode_summarization_agent)
        self.episode_summarization_agent.add_message(summary, role="assistant")
        return summary

    def summarize_narrative(self, trajectory):
        self.narrative_summarization_agent.add_message(trajectory, role="user")
        return call_llm_safe(self.narrative_summarization_agent)

    def _generate_step_by_step_plan(
        self,
        observation: Dict,
        instruction: str,
        failed_subtask: Optional[Node] = None,
        completed_subtasks_list: List[Node] = [],
        remaining_subtasks_list: List[Node] = [],
    ) -> Tuple[Dict, str]:
        agent = self.grounding_agent

        def format_subtask_list(subtasks: List[Node]) -> str:
            text = ""
            for index, node in enumerate(subtasks):
                text += f"{index + 1}. **{node.name}**:\n"
                for sentence in re.split(r"(?<=[.!?;]) +", node.info):
                    text += f"   - {sentence}\n"
                text += "\n"
            return text

        if self.turn_count == 0:
            self.search_query = self.knowledge_base.formulate_query(
                instruction, observation
            )

            similar_task = ""
            experience = ""
            integrated_knowledge = ""

            similar_task, experience = (
                self.knowledge_base.retrieve_narrative_experience(instruction)
            )
            logger.info(
                "SIMILAR TASK EXPERIENCE: %s",
                similar_task + "\n" + experience.strip(),
            )

            if self.search_engine is not None:
                web_knowledge = self.knowledge_base.retrieve_knowledge(
                    instruction=instruction,
                    search_query=self.search_query,
                    search_engine=self.search_engine,
                )
                logger.info("RETRIEVED KNOWLEDGE: %s", web_knowledge)

                if web_knowledge is not None:
                    integrated_knowledge = self.knowledge_base.knowledge_fusion(
                        observation=observation,
                        instruction=instruction,
                        web_knowledge=web_knowledge,
                        similar_task=similar_task,
                        experience=experience,
                    )
                    logger.info("INTEGRATED KNOWLEDGE: %s", integrated_knowledge)

            integrated_knowledge = integrated_knowledge or experience
            if integrated_knowledge:
                instruction += (
                    "\nYou may refer to some retrieved knowledge if you think they "
                    f"are useful.{integrated_knowledge}"
                )

            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
            )

        if failed_subtask:
            message = (
                f"The subtask {failed_subtask} cannot be completed. Please generate "
                "a new plan for the remainder of the trajectory.\n\n"
                "Successfully Completed Subtasks:\n"
                f"{format_subtask_list(completed_subtasks_list)}\n"
            )
        elif len(completed_subtasks_list) + len(remaining_subtasks_list) > 0:
            message = (
                "The current trajectory and desktop state is provided. Please revise "
                "the plan for the following trajectory.\n\n"
                "Successfully Completed Subtasks:\n"
                f"{format_subtask_list(completed_subtasks_list)}\n"
                "Future Remaining Subtasks:\n"
                f"{format_subtask_list(remaining_subtasks_list)}\n"
            )
        else:
            message = "Please generate the initial plan for the task.\n"

        logger.info("GENERATOR MESSAGE: %s", message)

        self.generator_agent.add_message(
            message,
            image_content=observation.get("screenshot", None),
            role="user",
        )

        logger.info("GENERATING HIGH LEVEL PLAN")
        plan = call_llm_safe(self.generator_agent)

        if plan == "":
            raise Exception("Plan Generation Failed - Fix the Prompt")

        logger.info("HIGH LEVEL STEP BY STEP PLAN: %s", plan)

        self.generator_agent.add_message(plan, role="assistant")
        self.planner_history.append(plan)
        self.turn_count += 1

        input_tokens, output_tokens = calculate_tokens(self.generator_agent.messages)
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)

        planner_info = {
            "search_query": self.search_query,
            "goal_plan": plan,
            "num_input_tokens_plan": input_tokens,
            "num_output_tokens_plan": output_tokens,
            "goal_plan_cost": cost,
        }

        assert type(plan) == str
        return planner_info, plan

    def _generate_dag(self, instruction: str, plan: str) -> Tuple[Dict, Dag]:
        self.dag_translator_agent.reset()

        self.dag_translator_agent.add_message(
            f"Instruction: {instruction}\nPlan: {plan}",
            role="user",
        )

        logger.info("GENERATING DAG")
        dag_response = call_llm_safe(self.dag_translator_agent)

        if dag_response == "":
            raise Exception("DAG Generation Failed - Fix the Prompt")

        logger.info("DAG: %s", dag_response)

        self.dag_translator_agent.add_message(dag_response, role="assistant")

        input_tokens, output_tokens = calculate_tokens(
            self.dag_translator_agent.messages
        )
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)

        dag_info = {
            "dag": dag_response,
            "num_input_tokens_dag": input_tokens,
            "num_output_tokens_dag": output_tokens,
            "dag_cost": cost,
        }

        assert type(dag_response) == str
        return dag_info, parse_dag(dag_response)