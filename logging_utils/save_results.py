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
    contextual_results=None,
    results_path="qna_eval/results/evaluation_results.json"
):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Load existing results with format filtering
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                existing_results = [
                    entry for entry in existing_results
                    if "model_name" in entry and "metrics" in entry
                ]
            except json.JSONDecodeError:
                print("⚠️ Corrupt or invalid JSON file. Starting fresh.")
                existing_results = []
    else:
        existing_results = []

    # Trim unwanted BLEU fields
    bleu_trimmed = {
        "bleu": bleu_results["bleu"],
        "precisions": bleu_results["precisions"]
    }

    # Build new entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "retrieval_method": retrieval_method,
        "dataset_name": dataset_name,
        "metrics": {
            "squad": squad_results,
            "rouge": rouge_results,
            "bleu": bleu_trimmed,
            "retrieval": mean_metrics
        }
    }

    if contextual_results:
        new_entry["metrics"]["contextual"] = contextual_results

    # Check if the new entry already exists in the results (avoid duplicates)
        existing_entry = next((
        entry for entry in existing_results
        if entry["model_name"] == model_name
        and entry["dataset_name"] == dataset_name
        and entry["retrieval_method"] == retrieval_method
        and entry["timestamp"] == timestamp
    ), None)

    if existing_entry is None:
        existing_results.append(new_entry)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=4)
        print(f"✅ Saved evaluation results to {results_path}")
    else:
        print("⚠️ Duplicate evaluation detected, not saving.")
