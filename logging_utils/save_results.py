import os
import json
from datetime import datetime

def save_evaluation_results(
    model_name,
    retrieval_method,
    dataset_name,
    squad_results,
    rouge_results,
    bleu_results,
    mean_metrics,
    path="results/evaluation_results.json"
):
    results = {
        "model": model_name,
        "retrieval_method": retrieval_method,
        "dataset": dataset_name,
        "metrics": {
            "exact_match": squad_results["exact_match"],
            "f1_score": squad_results["f1"],
            "rouge_l": rouge_results["rouge1"],
            "bleu": bleu_results["bleu"],
            "MAP": mean_metrics["map"],
            "NDCG": mean_metrics["ndcg"],
            "MRR": mean_metrics["recip_rank"]
        },
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    existing_results.append(results)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    print(f"\n Saved results to {path}")
