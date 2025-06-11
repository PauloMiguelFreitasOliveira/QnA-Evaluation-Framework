from transformers import pipeline
from datasets import load_dataset
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from rank_bm25 import BM25Okapi
import pytrec_eval
import torch
import numpy as np
import json
import sys
import os

# Adds the parent directory to the Python path so it can find logging_utils
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
from logging_utils.save_results import save_evaluation_results



dataset_path = os.path.join(ROOT_DIR, "datasets", "ms_marco.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    ms_marco_data = json.load(f)

NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", 10000))

ms_marco_data = ms_marco_data[:NUM_EXAMPLES]

query_passage_pairs = []

for entry in ms_marco_data:
    query_passage_pairs.append({     
        "query_id": entry["query_id"],
        "query": entry["query"],
        "answers": entry["answers"],
        "candidate_passages": [c["passage_text"] for c in entry["candidates"]],
        "is_selected": [c["is_selected"] for c in entry["candidates"]]
    })


def bm25_rerank(query, passages):
    tokenized_passages = [p.split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    return scores

model_name = "distilbert-base-uncased-distilled-squad"
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("question-answering", model=model_name, device=device)

squad_metric = evaluate.load("squad")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

references = []
predictions = []
qrel = {}
run = {}



# Evaluation loop
for entry in query_passage_pairs:
    question = entry["query"]
    passages = entry["candidate_passages"]
    query_id = str(entry["query_id"])
    true_answer = entry["answers"][0] if entry["answers"] else ""

    bm25_scores = bm25_rerank(question, passages)

    # Pick the top passage based on BM25 score
    best_idx = np.argmax(bm25_scores)
    top_context = passages[best_idx]

    if query_id == "some_specific_query_id":
        print("\nQuestion:", question)
        ranked = sorted(zip(passages, bm25_scores), key=lambda x: x[1], reverse=True)
        for i, (p, score) in enumerate(ranked[:5]):
            print(f"Rank {i+1} | Score: {score:.2f} | Passage: {p[:100]}")


    # Get model answer
    result = qa_pipeline(question=question, context=top_context)
    predicted_answer = result["answer"]

    # Append for metric calculation
    references.append({"id": query_id, "answers": {"text": entry["answers"], "answer_start": [0]}})
    predictions.append({"id": query_id, "prediction_text": predicted_answer})

    # Build qrel and run for pytrec_eval (top10 ranking)
    qrel[query_id] = {str(i): entry["is_selected"][i] for i in range(len(passages))}
    run[query_id] = {str(i): float(bm25_scores[i]) for i in range(len(passages))}

assert len(bm25_scores) == len(passages)
assert not any(np.isnan(bm25_scores))


# Metric results
squad_results = squad_metric.compute(predictions=predictions, references=references)
rouge_results = rouge_metric.compute(
    predictions=[pred["prediction_text"] for pred in predictions],
    references=[ref["answers"]["text"][0] for ref in references]
)
bleu_results = bleu_metric.compute(
    predictions=[pred["prediction_text"] for pred in predictions],
    references=[[ref["answers"]["text"][0]] for ref in references]
)

print(f"\n🎯 QA Metrics")
print(f"Exact Match: {squad_results['exact_match']}")
print(f"F1 Score: {squad_results['f1']}")
print(f"ROUGE-l F1 Score: {rouge_results['rouge1'] :.4f}")
print(f"BLEU Score: {bleu_results['bleu']}")

# Retrieval metrics
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'recip_rank'})
retrieval_metrics = evaluator.evaluate(run)

mean_metrics = {
    metric: np.mean([query_measures[metric] for query_measures in retrieval_metrics.values()])
    for metric in ['map', 'ndcg', 'recip_rank']
}

print(f"\n📊 BM25 Ranking Evaluation")
print(f"Mean Average Precision (MAP):     {mean_metrics['map']:.4f}")
print(f"Normalized DCG (NDCG):            {mean_metrics['ndcg']:.4f}")
print(f"Mean Reciprocal Rank (MRR):       {mean_metrics['recip_rank']:.4f}")    

save_evaluation_results(
    model_name=model_name,
    evaluation_method="BM25",
    dataset_name="ms_marco",
    squad_results=squad_results,
    rouge_results=rouge_results,
    bleu_results=bleu_results,
    mean_metrics=mean_metrics
)

query = "capital of France"
passages = [
    "Paris is the capital of France.",
    "France has many beautiful cities including Lyon and Marseille.",
    "The Eiffel Tower is a famous landmark."
]

scores = bm25_rerank(query, passages)

print(scores)

