from rank_bm25 import BM25Okapi
import numpy as np

def is_relevant(passage, answers):
    return any(ans.lower() in passage.lower() for ans in answers)

def tokenize(text):
    return text.lower().split()

def retrieve_top_k_bm25(queries, top_k, context_pool=None):
    assert context_pool is not None, "BM25 requires a context pool."
    
    tokenized_corpus = [tokenize(doc) for doc in context_pool]
    bm25 = BM25Okapi(tokenized_corpus)

    results = []
    for item in queries:
        query = item["query"]
        query_tokens = tokenize(query)

        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_passages = [context_pool[i] for i in top_indices]

        candidates = [
            {"passage_text": context_pool[i], "is_selected": int(is_relevant(context_pool[i], item["answers"]))}
            for i in range(len(context_pool))
        ]

        top_relevance = [
            {
                "text": context_pool[i],
                "score": float(scores[i]),
                "is_selected": int(is_relevant(context_pool[i], item["answers"]))
            }
            for i in top_indices
        ]

        results.append({
            "query": query,
            "query_id": item["query_id"],
            "answers": item["answers"],
            "top_passages": top_passages,
            "scores": [float(scores[i]) for i in top_indices],
            "top_relevance": top_relevance,
            "qrel": {str(i): int(candidates[i]["is_selected"]) for i in range(len(context_pool))},
            "run": {str(i): float(scores[i]) for i in range(len(context_pool))}
        })

    return results
