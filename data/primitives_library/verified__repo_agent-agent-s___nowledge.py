import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gui_agents.s1.core.BaseModule import BaseModule
from gui_agents.s1.core.ProceduralMemory import PROCEDURAL_MEMORY
from gui_agents.s1.mllm.MultimodalEngine import OpenAIEmbeddingEngine
from gui_agents.s1.utils.common_utils import (
    load_embeddings,
    load_knowledge_base,
    save_embeddings,
)
from gui_agents.s1.utils.query_perplexica import query_to_perplexica


class KnowledgeBase(BaseModule):
    def __init__(
        self,
        local_kb_path: str,
        platform: str,
        engine_params: Dict,
        use_image_for_search: bool = False,
    ):
        super().__init__(engine_params, platform)

        self.local_kb_path = local_kb_path
        api_key = engine_params["api_key"] if "api_key" in engine_params else os.getenv(
            "OPENAI_API_KEY"
        )
        self.embedding_engine = OpenAIEmbeddingEngine(api_key=api_key)

        platform_path = os.path.join(self.local_kb_path, self.platform)
        self.episodic_memory_path = os.path.join(
            platform_path, "episodic_memory.json"
        )
        self.narrative_memory_path = os.path.join(
            platform_path, "narrative_memory.json"
        )
        self.embeddings_path = os.path.join(platform_path, "embeddings.pkl")

        self.rag_module_system_prompt = PROCEDURAL_MEMORY.RAG_AGENT.replace(
            "CURRENT_OS", self.platform
        )
        self.query_formulator = self._create_agent(self.rag_module_system_prompt)
        self.llm_search_agent = self._create_agent(self.rag_module_system_prompt)
        self.knowledge_fusion_agent = self._create_agent(
            self.rag_module_system_prompt
        )

        self.use_image_for_search = use_image_for_search

    def retrieve_knowledge(
        self, instruction: str, search_query: str, search_engine: str = "llm"
    ) -> Tuple[str, str]:
        search_result = self._search(instruction, search_query, search_engine)
        return search_query, search_result

    def formulate_query(self, instruction: str, observation: Dict) -> str:
        query_file = os.path.join(
            self.local_kb_path, self.platform, "formulate_query.json"
        )

        try:
            with open(query_file, "r") as file:
                cached_queries = json.load(file)
        except:
            cached_queries = {}

        if instruction in cached_queries:
            return cached_queries[instruction]

        screenshot = None
        if self.use_image_for_search and "screenshot" in observation:
            screenshot = observation["screenshot"]

        self.query_formulator.add_message(
            f"The task is: {instruction}\n"
            f"Accessibility tree of the current desktop UI state: "
            f"{observation['linearized_accessibility_tree']}\n"
            "To use google search to get some useful information, first carefully analyze "
            "the accessibility tree of the current desktop UI state, then given the task "
            "instruction, formulate a question that can be used to search on the Internet "
            "for information in helping with the task execution.\n"
            "The question should not be too general or too specific. Please ONLY provide "
            "the question.\nQuestion:",
            image_content=screenshot,
        )

        query = self.query_formulator.get_response().strip().replace('"', "")
        print("search query: ", query)

        cached_queries[instruction] = query
        with open(query_file, "w") as file:
            json.dump(cached_queries, file, indent=2)

        return query

    def _search(self, instruction: str, search_query: str, search_engine: str) -> str:
        cache_path = os.path.join(
            self.local_kb_path,
            self.platform,
            f"{search_engine}_rag_knowledge.json",
        )

        try:
            with open(cache_path, "r") as file:
                cached_results = json.load(file)
        except:
            cached_results = {}

        if instruction in cached_results:
            return cached_results[instruction]

        if search_engine.lower() == "llm":
            self.llm_search_agent.add_message(search_query)
            result = self.llm_search_agent.get_response()
        elif search_engine.lower() == "perplexica":
            result = query_to_perplexica(search_query)
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")

        cached_results[instruction] = result.strip()
        with open(cache_path, "w") as file:
            json.dump(cached_results, file, indent=2)

        return result

    def retrieve_narrative_experience(self, instruction: str) -> Tuple[str, str]:
        knowledge_base = load_knowledge_base(self.narrative_memory_path)
        if not knowledge_base:
            return "None", "None"

        embeddings = load_embeddings(self.embeddings_path)
        instruction_embedding = embeddings.get(instruction)

        if instruction_embedding is None:
            instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            embeddings[instruction] = instruction_embedding

        candidate_embeddings = []
        for task in knowledge_base:
            task_embedding = embeddings.get(task)
            if task_embedding is None:
                task_embedding = self.embedding_engine.get_embeddings(task)
                embeddings[task] = task_embedding
            candidate_embeddings.append(task_embedding)

        save_embeddings(self.embeddings_path, embeddings)

        scores = cosine_similarity(
            instruction_embedding, np.vstack(candidate_embeddings)
        )[0]
        ranking = np.argsort(scores)[::-1]
        tasks = list(knowledge_base.keys())

        selected_index = 1 if tasks[ranking[0]] == instruction else 0
        selected_task = tasks[ranking[selected_index]]
        return selected_task, knowledge_base[selected_task]

    def retrieve_episodic_experience(self, instruction: str) -> Tuple[str, str]:
        knowledge_base = load_knowledge_base(self.episodic_memory_path)
        if not knowledge_base:
            return "None", "None"

        embeddings = load_embeddings(self.embeddings_path)
        instruction_embedding = embeddings.get(instruction)

        if instruction_embedding is None:
            instruction_embedding = self.embedding_engine.get_embeddings(instruction)
            embeddings[instruction] = instruction_embedding

        candidate_embeddings = []
        for task in knowledge_base:
            task_embedding = embeddings.get(task)
            if task_embedding is None:
                task_embedding = self.embedding_engine.get_embeddings(task)
                embeddings[task] = task_embedding
            candidate_embeddings.append(task_embedding)

        save_embeddings(self.embeddings_path, embeddings)

        scores = cosine_similarity(
            instruction_embedding, np.vstack(candidate_embeddings)
        )[0]
        ranking = np.argsort(scores)[::-1]
        tasks = list(knowledge_base.keys())

        selected_index = 1 if tasks[ranking[0]] == instruction else 0
        selected_task = tasks[ranking[selected_index]]
        return selected_task, knowledge_base[selected_task]

    def knowledge_fusion(
        self,
        observation: Dict,
        instruction: str,
        web_knowledge: str,
        similar_task: str,
        experience: str,
    ) -> str:
        self.knowledge_fusion_agent.add_message(
            f"The task is: {instruction}\n"
            f"Accessibility tree of the current desktop UI state: "
            f"{observation['linearized_accessibility_tree']}\n"
            f"Web knowledge: {web_knowledge}\n"
            f"Similar task: {similar_task}\n"
            f"Experience: {experience}\n"
            "Based on the above information, provide useful information that can help "
            "with task execution. Focus on the most relevant and actionable details.",
            image_content=(
                observation["screenshot"]
                if self.use_image_for_search and "screenshot" in observation
                else None
            ),
        )
        return self.knowledge_fusion_agent.get_response()