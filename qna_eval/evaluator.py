import pytrec_eval
import numpy as np
from .extractive_eval import evaluate_extractive_model
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a model once globally
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_qa(queries, predictions, model_name):
    gt = {q["query_id"]: q["answers"] for q in queries}
    return evaluate_extractive_model(predictions, gt, model_name)

def evaluate_retrieval(qrel, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'recip_rank'})
    retrieval_metrics = evaluator.evaluate(run)

    mean_metrics = {
        metric: np.mean([query_measures[metric] for query_measures in retrieval_metrics.values()])
        for metric in ['map', 'ndcg', 'recip_rank']
    }

    return mean_metrics

# ------------------ CONTEXTUAL METRICS ------------------

def contextual_precision(relevant_indices, retrieved):
    if not retrieved:
        return 0.0
    precision_scores = [1 if i in relevant_indices else 0 for i in range(len(retrieved))]
    return sum(precision_scores) / len(precision_scores)

def contextual_recall(ground_truth_statements, retrieved_chunks):
    if not ground_truth_statements:
        return 0.0

    count = 0
    for statement in ground_truth_statements:
        if any(statement.lower() in chunk.lower() for chunk in retrieved_chunks):
            count += 1
    return count / len(ground_truth_statements)

def contextual_relevancy(query, retrieved_chunks):
    if not retrieved_chunks:
        return 0.0
    query_embedding = semantic_model.encode([query])
    context_embeddings = semantic_model.encode(retrieved_chunks)
    similarities = cosine_similarity(query_embedding, context_embeddings)[0]
    return float(np.mean(similarities))

def evaluate_contextual(retrieval_results):
    contextual_precisions = []
    contextual_recalls = []
    contextual_relevancies = []

    for result in retrieval_results:
        top_relevance_scores = result.get("top_relevance", [])

        if not top_relevance_scores:
            continue  # skip if missing relevance

        num_relevant = sum(1 for entry in top_relevance_scores if entry["is_selected"] > 0)
        total_passages = len(top_relevance_scores)
        relevant_in_all = sum(1 for entry in top_relevance_scores if entry["is_selected"] == 1)



        precision = num_relevant / total_passages if total_passages else 0.0
        recall = num_relevant / 1.0  # each query has at least one relevant passage
        relevancy = relevant_in_all / total_passages if total_passages else 0.0

        contextual_precisions.append(precision)
        contextual_recalls.append(recall)
        contextual_relevancies.append(relevancy)

    return {
        "contextual_precision": round(sum(contextual_precisions) / len(contextual_precisions), 4) if contextual_precisions else 0.0,
        "contextual_recall": round(sum(contextual_recalls) / len(contextual_recalls), 4) if contextual_recalls else 0.0,
        "contextual_relevancy": round(sum(contextual_relevancies) / len(contextual_relevancies), 4) if contextual_relevancies else 0.0
    }
