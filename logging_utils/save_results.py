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
    results_path="results/evaluation_results.json"
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

    if new_entry not in existing_results:
        existing_results.append(new_entry)
    else:
        print("⚠️ Duplicate evaluation, not saving.")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=4)

    print(f"✅ Saved evaluation results to {results_path}")
