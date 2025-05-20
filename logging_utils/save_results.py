import os
import json
from datetime import datetime

def save_evaluation_results(
    model_name,
    evaluation_method,
    dataset_name,
    squad_results,
    rouge_results,
    bleu_results,
    mean_metrics,
    contextual_results=None,
    truth_metrics=None,
    examples=None,  
    num_entries=None,
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
         "evaluation_method": evaluation_method,
         "dataset_name": dataset_name,
         "num_entries": num_entries,
         "metrics": {
             "squad": squad_results,
             "rouge": rouge_results,
             "bleu": bleu_trimmed,
             "retrieval": mean_metrics
         }
     }  
     # Add contextual metrics if provided
    if contextual_results:
        new_entry["metrics"]["contextual"] = contextual_results    
    # Add TruthfulQA metrics if provided
    if truth_metrics:
        new_entry["metrics"]["truthfulqa"] = truth_metrics    
    if examples:
        new_entry["examples"] = examples    
    # Check if the new entry already exists in the results (avoid duplicates)
    existing_entry = next((
        entry for entry in existing_results
        if entry["model_name"] == model_name
        and entry["dataset_name"] == dataset_name
        and entry["evaluation_method"] == evaluation_method
        and entry["timestamp"] == timestamp
    ), None)

    if existing_entry is None:
        existing_results.append(new_entry)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=4)
        print(f"✅ Saved evaluation results to {results_path}")
    else:
        print("⚠️ Duplicate evaluation detected, not saving.")

        print("\n---- Evaluation Summary ----")
    print(f"Model:      {model_name}")
    print(f"Dataset:    {dataset_name}")
    print(f"Method:     {evaluation_method}")
    print(f"Entries run:{num_entries}")
    print("\n📊 Squad Metrics:")
    print(f"  Exact Match: {squad_results.get('exact_match')}")
    print(f"  F1:          {squad_results.get('f1')}")
    print("\n📊 Rouge Metrics:")
    for k, v in rouge_results.items():
        print(f"  {k}: {v}")
    print(f"\n📊 BLEU: {bleu_trimmed['bleu']}")
    if mean_metrics:
        print("\n📊 Retrieval Metrics:")
        for k,v in mean_metrics.items():
            print(f"  {k}: {v}")
    if contextual_results:
        print("\n📊 Contextual Metrics:")
        for k,v in contextual_results.items():
            print(f"  {k}: {v}")
    if truth_metrics:
        print("\n📊 TruthfulQA (hallucination) Metrics:")
        for k,v in truth_metrics.items():
            print(f"  {k}: {v}")
    print("------------------------------\n")