# qna_eval/retriever_bm25.py

from rank_bm25 import BM25Okapi
import numpy as np

def is_relevant(passage, answers):
    return any(ans.lower() in passage.lower() for ans in answers)

def tokenize(text):
    return text.lower().split()

def retrieve_top_k_bm25(queries, top_k, context_pool=None):
    results = []

    # If there is a global pool, build one BM25 index and reuse it
    if context_pool:
        global_tokenized = [tokenize(doc) for doc in context_pool]
        global_bm25 = BM25Okapi(global_tokenized)

        for item in queries:
            qid     = item["query_id"]
            query   = item["query"]
            answers = item.get("answers", [])

            q_tokens     = tokenize(query)
            scores       = global_bm25.get_scores(q_tokens)
            top_indices  = np.argsort(scores)[::-1][:top_k]

            top_passages = [context_pool[i] for i in top_indices]
            top_relevance = [
                {
                    "text":       context_pool[i],
                    "score":      float(scores[i]),
                    "is_selected": int(is_relevant(context_pool[i], answers))
                }
                for i in top_indices
            ]
            # For global qrel/run we still report over the entire pool
            qrel = {str(i): int(is_relevant(context_pool[i], answers))
                    for i in range(len(context_pool))}
            run  = {str(i): float(scores[i])
                    for i in range(len(context_pool))}

            results.append({
                "query":         query,
                "query_id":      qid,
                "answers":       answers,
                "top_passages":  top_passages,
                "scores":        [float(scores[i]) for i in top_indices],
                "top_relevance": top_relevance,
                "qrel":          qrel,
                "run":           run
            })

    # Otherwise, do BM25 separately on each query’s own candidates
    else:
        for item in queries:
            qid     = item["query_id"]
            query   = item["query"]
            answers = item.get("answers", [])

            # pull that query's passages
            candidate_texts = [
                c.get("passage_text", "") for c in item.get("candidates", [])
            ]
            if not candidate_texts:
                # can't retrieve anything if there are no candidates
                continue

            tokenized = [tokenize(doc) for doc in candidate_texts]
            bm25 = BM25Okapi(tokenized)

            q_tokens    = tokenize(query)
            scores      = bm25.get_scores(q_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]

            top_passages = [candidate_texts[i] for i in top_indices]
            top_relevance = [
                {
                    "text":       candidate_texts[i],
                    "score":      float(scores[i]),
                    "is_selected": int(is_relevant(candidate_texts[i], answers))
                }
                for i in top_indices
            ]
            # qrel/run only over this candidate list
            qrel = {str(i): int(is_relevant(candidate_texts[i], answers))
                    for i in range(len(candidate_texts))}
            run  = {str(i): float(scores[i])
                    for i in range(len(candidate_texts))}

            results.append({
                "query":         query,
                "query_id":      qid,
                "answers":       answers,
                "top_passages":  top_passages,
                "scores":        [float(scores[i]) for i in top_indices],
                "top_relevance": top_relevance,
                "qrel":          qrel,
                "run":           run
            })

    return results
