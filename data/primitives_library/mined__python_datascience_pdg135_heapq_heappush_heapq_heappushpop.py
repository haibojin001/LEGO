# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg135::heapq.heappush+heapq.heappushpop
# name: heapq_primitive
# summary: Uses heapq.heappush, heapq.heappushpop across 2 repos
# anchor_symbols: ['heapq.heappush', 'heapq.heappushpop']
# observed in 2 repos: ['microsoft__nni', 'refuel-ai__autolabel']...

# --- from microsoft__nni::examples/trials/weight_sharing/ga_squad/trial.py::MaxQueue.push ---
def push(self, item):
        if self.size < self.capacity:
            heapq.heappush(self.entries, item)
        else:
            heapq.heappushpop(self.entries, item)

# --- from microsoft__nni::examples/trials/ga_squad/trial.py::MaxQueue.push ---
def push(self, item):
        if self.size < self.capacity:
            heapq.heappush(self.entries, item)
        else:
            heapq.heappushpop(self.entries, item)

# --- from refuel-ai__autolabel::src/autolabel/few_shot/vector_store.py::semantic_search ---
def semantic_search(
    query_embeddings,
    corpus_embeddings,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function=cos_sim,
):
    """
    Semantic similarity search based on cosine similarity score. Implementation from this project: https://github.com/UKPLab/sentence-transformers
    """
    if isinstance(query_embeddings, list):
        query_embeddings = np.array(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.reshape(1, -1)

    if isinstance(corpus_embeddings, list):
        corpus_embeddings = np.array(corpus_embeddings)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[
                    corpus_start_idx : corpus_start_idx + corpus_chunk_size
                ],
            )

            # Get top-k scores
            cos_scores_top_k_values = np.sort(cos_scores, axis=1)[:, -top_k:][:, ::-1]
            cos_scores_top_k_idx = np.argsort(cos_scores, axis=1)[:, -top_k:][:, ::-1]
            cos_scores_top_k_values = cos_scores_top_k_values.tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr],
                ):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id),
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(
                            queries_result_list[query_id], (score, corpus_id),
                        )

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {
                "corpus_id": corpus_id,
                "score": score,
            }
        queries_result_list[query_id] = sorted(
            queries_result_list[query_id], key=lambda x: x["score"], reverse=True,
        )
    return queries_result_list
