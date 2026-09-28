import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gui_agents.s2.core.module import BaseModule
from gui_agents.s2.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2.utils.common_utils import (
    call_llm_safe,
    load_embeddings,
    load_knowledge_base,
    save_embeddings,
)
from gui_agents.s2.utils.query_perplexica import query_to_perplexica


class KnowledgeBase(BaseModule):
    def __init__(
        self,
        embedding_engine,
        local_kb_path: str,
        platform: str,
        engine_params: Dict,
        save_knowledge: bool = True,
    ):
        super().__init__(engine_params, platform)

        self.local_kb_path = local_kb_path
        self.embedding_engine = embedding_engine

        self.episodic_memory_path = os.path.join(
            self.local_kb_path, self.platform, "episodic_memory.json"
        )
        self.narrative_memory_path = os.path.join(
            self.local_kb_path, self.platform, "narrative_memory.json"
        )
        self.embeddings_path = os.path.join(
            self.local_kb_path, self.platform, "embeddings.pkl"
        )

        self.task_trajectory = ""
        self.current_subtask_trajectory = ""
        self.current_search_query = ""

        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT.replace(
            "CURRENT_OS", self.platform
        )

        self.query_formulator = self._create_agent(self.rag_module_system_prompt)
        self.llm_search_agent = self._create_agent(self.rag_module_system_prompt)
        self.knowledge_fusion_agent = self._create_agent(
            self.rag_module_system_prompt
        )

        self.narrative_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.TASK_SUMMARIZATION_PROMPT
        )
        self.episode_summarization_agent = self._create_agent(
            PROCEDURAL_MEMORY.SUBTASK_SUMMARIZATION_PROMPT
        )

        self.save_knowledge = save_knowledge

    def retrieve_knowledge(
        self, instruction: str, search_query: str, search_engine: str = "llm"
    ) -> Tuple[str, str]:
        search_results = self._search(instruction, search_query, search_engine)
        return search_query, search_results

    def formulate_query(self, instruction: str, observation: Dict) -> str:
        query_path = os.path.join(
            self.local_kb_path, self.platform, "formulate_query.json"
        )

        try:
            with open(query_path, "r") as file:
                cached_queries = json.load(file)
        except Exception:
            cached_queries = {}

        if instruction in cached_queries:
            return cached_queries[instruction]

        self.query_formulator.reset()
        self.query_formulator.add_message(
            f"The task is: {instruction}\n"
            "To use google search to get some useful information, first carefully analyze "
            "the screenshot of the current desktop UI state, then given the task "
            "instruction, formulate a question that can be used to search on the Internet "
            "for information in helping with the task execution.\n"
            "The question should not be too general or too specific. Please ONLY provide "
            "the question.\nQuestion:",
            image_content=(
                observation["screenshot"] if "screenshot" in observation else None
            ),
            role="user",
        )

        search_query = self.query_formulator.get_response().strip().replace('"', "")
        print("search query: ", search_query)

        cached_queries[instruction] = search_query
        with open(query_path, "w") as file:
            json.dump(cached_queries, file, indent=2)

        return search_query

    def _search(self, instruction: str, search_query: str, search_engine: str) -> str:
        knowledge_path = os.path.join(
            self.local_kb_path,
            self.platform,
            f"{search_engine}_rag_knowledge.json",
        )

        try:
            with open(knowledge_path, "r") as file:
                cached_results = json.load(file)
        except Exception:
            cached_results = {}

        if instruction in cached_results:
            return cached_results[instruction]

        if search_engine.lower() == "llm":
            self.llm_search_agent.reset()
            self.llm_search_agent.add_message(search_query, role="user")
            search_results = self.llm_search_agent.get_response()
        elif search_engine.lower() == "perplexica":
            search_results = query_to_perplexica(search_query)
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")

        cached_results[instruction] = search_results.strip()
        with open(knowledge_path, "w") as file:
            json.dump(cached_results, file, indent=2)

        return search_results

    def _retrieve_experience(self, instruction: str, knowledge_path: str) -> Tuple[str, str]:
        knowledge_base = load_knowledge_base(knowledge_path)
        if not knowledge_base:
            return "None", "None"

        embeddings = load_embeddings(self.embeddings_path)

        instruction_embedding = embeddings.get(instruction)
        if instruction_embedding is None:
            instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            embeddings[instruction] = instruction_embedding

        candidate_embeddings = []
        for key in knowledge_base:
            candidate_embedding = embeddings.get(key)
            if candidate_embedding is None:
                candidate_embedding = self.embedding_engine.get_embeddings(key)
                embeddings[key] = candidate_embedding
            candidate_embeddings.append(candidate_embedding)

        save_embeddings(self.embeddings_path, embeddings)

        similarities = cosine_similarity(
            instruction_embedding, np.vstack(candidate_embeddings)
        )[0]
        sorted_indices = np.argsort(similarities)[::-1]
        keys = list(knowledge_base.keys())

        index = 1 if keys[sorted_indices[0]] == instruction else 0
        selected_key = keys[sorted_indices[index]]
        return selected_key, knowledge_base[selected_key]

    def retrieve_narrative_experience(self, instruction: str) -> Tuple[str, str]:
        return self._retrieve_experience(instruction, self.narrative_memory_path)

    def retrieve_episodic_experience(self, instruction: str) -> Tuple[str, str]:
        return self._retrieve_experience(instruction, self.episodic_memory_path)

    def knowledge_fusion(
        self,
        observation: Dict,
        instruction: str,
        retrieved_knowledge: str = "None",
        narrative_experience: str = "None",
        episodic_experience: str = "None",
    ) -> str:
        self.knowledge_fusion_agent.reset()

        message = (
            f"Task instruction:\n{instruction}\n\n"
            f"Retrieved web knowledge:\n{retrieved_knowledge}\n\n"
            f"Narrative experience:\n{narrative_experience}\n\n"
            f"Episodic experience:\n{episodic_experience}\n\n"
            "Based on the current UI state, task instruction, and available knowledge, "
            "provide useful information and guidance for completing the task."
        )

        self.knowledge_fusion_agent.add_message(
            message,
            image_content=(
                observation["screenshot"] if "screenshot" in observation else None
            ),
            role="user",
        )
        return self.knowledge_fusion_agent.get_response()

    def update_task_trajectory(self, trajectory: str):
        self.task_trajectory += trajectory

    def update_subtask_trajectory(self, trajectory: str):
        self.current_subtask_trajectory += trajectory

    def reset_subtask_trajectory(self):
        self.current_subtask_trajectory = ""

    def _safe_response(self, agent):
        try:
            return call_llm_safe(agent)
        except TypeError:
            return agent.get_response()

    def summarize_episode(self, instruction: str) -> str:
        self.episode_summarization_agent.reset()
        self.episode_summarization_agent.add_message(
            f"Task instruction:\n{instruction}\n\n"
            f"Task trajectory:\n{self.task_trajectory}",
            role="user",
        )
        return self._safe_response(self.episode_summarization_agent)

    def summarize_narrative(self, instruction: str) -> str:
        self.narrative_summarization_agent.reset()
        self.narrative_summarization_agent.add_message(
            f"Task instruction:\n{instruction}\n\n"
            f"Subtask trajectory:\n{self.current_subtask_trajectory}",
            role="user",
        )
        return self._safe_response(self.narrative_summarization_agent)

    def save_episodic_memory(self, instruction: str, experience: str):
        if not self.save_knowledge:
            return

        knowledge_base = load_knowledge_base(self.episodic_memory_path)
        knowledge_base[instruction] = experience
        with open(self.episodic_memory_path, "w") as file:
            json.dump(knowledge_base, file, indent=2)

    def save_narrative_memory(self, instruction: str, experience: str):
        if not self.save_knowledge:
            return

        knowledge_base = load_knowledge_base(self.narrative_memory_path)
        knowledge_base[instruction] = experience
        with open(self.narrative_memory_path, "w") as file:
            json.dump(knowledge_base, file, indent=2)