import logging
import platform
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from gui_agents.s1.aci.ACI import ACI
from gui_agents.s1.core.BaseModule import BaseModule
from gui_agents.s1.core.Knowledge import KnowledgeBase
from gui_agents.s1.core.ProceduralMemory import PROCEDURAL_MEMORY
from gui_agents.s1.utils.common_utils import (
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
        search_engine: Optional[str] = None,
        multi_round: bool = False,
        platform: str = platform.system().lower(),
    ):
        super().__init__(engine_params, platform)

        self.grounding_agent = grounding_agent
        self.generator_agent = self._create_agent(PROCEDURAL_MEMORY.MANAGER_PROMPT)
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
        self.knowledge_base = KnowledgeBase(local_kb_path, platform, engine_params)

        self.planner_history = []
        self.turn_count = 0
        self.search_engine = search_engine
        self.multi_round = multi_round
        self.platform = platform

    def summarize_episode(self, trajectory):
        self.episode_summarization_agent.add_message(trajectory)
        summary = call_llm_safe(self.episode_summarization_agent)
        self.episode_summarization_agent.add_message(summary)
        return summary

    def summarize_narrative(self, trajectory):
        self.narrative_summarization_agent.add_message(trajectory)
        return call_llm_safe(self.narrative_summarization_agent)

    def _generate_step_by_step_plan(
        self, observation: Dict, instruction: str, failure_feedback: str = ""
    ) -> Tuple[Dict, str]:
        agent = self.grounding_agent
        self.active_apps = agent.get_active_apps(observation)

        accessibility_tree = agent.linearize_and_annotate_tree(observation)
        observation["linearized_accessibility_tree"] = accessibility_tree

        if self.turn_count == 0:
            self.search_query = self.knowledge_base.formulate_query(
                instruction, observation
            )

            similar_task, experience = (
                self.knowledge_base.retrieve_narrative_experience(instruction)
            )
            logger.info(
                "SIMILAR TASK EXPERIENCE: %s",
                similar_task + "\n" + experience.strip(),
            )

            merged_knowledge = ""
            if self.search_engine is not None:
                web_knowledge = self.knowledge_base.retrieve_knowledge(
                    instruction=instruction,
                    search_query=self.search_query,
                    search_engine=self.search_engine,
                )
                logger.info("RETRIEVED KNOWLEDGE: %s", web_knowledge)

                if web_knowledge is not None:
                    merged_knowledge = self.knowledge_base.knowledge_fusion(
                        observation=observation,
                        instruction=instruction,
                        web_knowledge=web_knowledge,
                        similar_task=similar_task,
                        experience=experience,
                    )
                    logger.info("INTEGRATED KNOWLEDGE: %s", merged_knowledge)

            merged_knowledge = merged_knowledge or experience
            if merged_knowledge:
                instruction += (
                    "\nYou may refer to some retrieved knowledge if you think they are useful."
                    + merged_knowledge
                )

            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
            )

        message = (
            f"Accessibility Tree: {accessibility_tree}\n"
            f"The clipboard contains: {agent.clipboard}."
            f"The current open applications are {agent.get_active_apps(observation)}"
        )
        if failure_feedback:
            message += f" Previous plan failed at step: {failure_feedback}"

        self.generator_agent.add_message(
            message, image_content=observation.get("screenshot", None)
        )

        logger.info("GENERATING HIGH LEVEL PLAN")
        plan = call_llm_safe(self.generator_agent)

        if plan == "":
            raise Exception("Plan Generation Failed - Fix the Prompt")

        logger.info("HIGH LEVEL STEP BY STEP PLAN: %s", plan)

        self.generator_agent.add_message(plan)
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
        self.dag_translator_agent.add_message(
            f"Instruction: {instruction}\nPlan: {plan}"
        )

        logger.info("GENERATING DAG")
        raw_dag = call_llm_safe(self.dag_translator_agent)
        dag = parse_dag(raw_dag)

        logger.info("Generated DAG: %s", raw_dag)

        self.dag_translator_agent.add_message(raw_dag)

        input_tokens, output_tokens = calculate_tokens(
            self.dag_translator_agent.messages
        )
        cost = input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)

        dag_info = {
            "dag": raw_dag,
            "num_input_tokens_dag": input_tokens,
            "num_output_tokens_dag": output_tokens,
            "dag_cost": cost,
        }

        assert type(dag) == Dag
        return dag_info, dag

    def _topological_sort(self, dag: Dag) -> List[Node]:
        def dfs(node_name, visited, stack):
            visited[node_name] = True
            for adjacent in adj_list[node_name]:
                if not visited[adjacent]:
                    dfs(adjacent, visited, stack)
            stack.append(node_name)

        adj_list = defaultdict(list)
        for source, destination in dag.edges:
            adj_list[source.name].append(destination.name)

        visited = {node.name: False for node in dag.nodes}
        stack = []

        for node in dag.nodes:
            if not visited[node.name]:
                dfs(node.name, visited, stack)

        sorted_nodes = [
            next(node for node in dag.nodes if node.name == name)
            for name in stack[::-1]
        ]
        return