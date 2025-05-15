import argparse
import os
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from qna_eval.dataset_loader import load_dataset_file
from qna_eval.extractive_eval import evaluate_extractive_model
from logging_utils.save_results import save_evaluation_results

def load_generator(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return pipe

def generate_predictions(pipe, queries, prompt_template="query: {question}", max_new_tokens=128):
    predictions = []

    for ex in queries:#tqdm(queries, desc="Generating answers"):
        prompt = prompt_template.format(question=ex.get("query", ""))
        output = pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]

        print(f"\nQuery ID: {ex['query_id']}")
        print(f"Question: {ex['query']}")
        print(f"Ideal Answers: {ex.get('answers', [])}")
        print(f"Model Answer: {output.strip()}")

        predictions.append({
            "query_id": ex["query_id"],
            "answer": output.strip()
        })

    return predictions

def build_ground_truth_dict(queries):
    return {ex["query_id"]: ex.get("answers", []) for ex in queries}

def main():
    parser = argparse.ArgumentParser(description="Run Generative QA Evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Generative model (e.g., flan-t5-base)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Just the dataset name (used for saving)")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, or CUDA device id (e.g., 0)")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of queries")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset_file(args.dataset_name, args.limit)
    queries = dataset["queries"]

    # Load model
    pipe = load_generator(args.model_name, args.device)

    # Generate answers
    predictions = generate_predictions(pipe, queries)


    # Evaluate
    ground_truth = build_ground_truth_dict(queries)
    metrics = evaluate_extractive_model(predictions, ground_truth)

    # Wrap for saving
    save_evaluation_results(
        model_name=args.model_name,
        retrieval_method="Generative",
        dataset_name=args.dataset_name,
        squad_results={
            "exact_match": metrics.get("exact_match", 0.0),
            "f1": metrics.get("f1", 0.0)
        },
        rouge_results={
            "rouge1": metrics.get("rouge1", 0.0),
            "rouge2": metrics.get("rouge2", 0.0),
            "rougeL": metrics.get("rougeL", 0.0),
            "rougeLsum": metrics.get("rougeLsum", 0.0)
        },
        bleu_results={
            "bleu": metrics.get("bleu", 0.0),
            "precisions": []  # Not used currently
        },
        mean_metrics={},  # Retrieval not applicable
        contextual_results=None,
        num_entries=len(queries)
    )

if __name__ == "__main__":
    main()
