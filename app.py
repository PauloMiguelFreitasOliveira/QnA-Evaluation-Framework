import argparse
import os
import json
from qna_eval.retriever import retrieve_top_k
from qna_eval.reader import load_reader, extract_answers
from qna_eval.dataset_loader import load_dataset_file
from qna_eval.evaluator import evaluate_retrieval, evaluate_qa
from logging_utils.save_results import save_evaluation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_model', required=True, help='Retriever model name or path')
    parser.add_argument('--reader_model', default=None, help='Reader (extractive QA) model name')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--top_k', type=int, default=10, help='Top K passages to retrieve')
    parser.add_argument('--limit', type=int, default=1000, help='Limit number of queries for testing')
    parser.add_argument('--max_context_per_query', type=int, default=100, help='Limit number of contextes per query')
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset_file(args.dataset, args.limit, args.max_context_per_query)
    queries = dataset["queries"]

    # Retrieve top-K passages (retriever logic is inside retrieve_top_k)
    results = retrieve_top_k(args.retriever_model, queries, top_k=args.top_k)

    # Prepare retrieval metrics inputs
    qrel = {r["query_id"]: r["qrel"] for r in results}
    run  = {r["query_id"]: r["run"]  for r in results}

    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(qrel, run)
    print("\n🔍 Retrieval Metrics:")
    print(retrieval_metrics)

    # Initialize QA metrics
    qa_metrics = {}
    if args.reader_model:
        # Load reader
        reader = load_reader(args.reader_model)
        
        predictions = extract_answers(reader, results, separator=" [SEP] ")


        # Evaluate QA (pass model name to evaluator)
        qa_metrics = evaluate_qa(queries, predictions, args.reader_model)

        print("\n🧠 QA Metrics:")
        print(qa_metrics)


    # Prepare arguments for save_results.py
    squad_results = {
        "exact_match": qa_metrics.get("exact_match", 0.0),
        "f1": qa_metrics.get("f1", 0.0)
    }
    rouge_results = {
        "rouge1": qa_metrics.get("rouge1", 0.0),
        "rouge2": qa_metrics.get("rouge2", 0.0),
        "rougeL": qa_metrics.get("rougeL", 0.0),
        "rougeLsum": qa_metrics.get("rougeLsum", 0.0)
    }
    bleu_results = {
        "bleu": qa_metrics.get("bleu", 0.0),
        "precisions": []  # not computed in this pipeline
    }

    # Save everything
    save_evaluation_results(
        model_name=args.reader_model,
        retrieval_method=args.retriever_model,
        dataset_name=args.dataset,
        squad_results=squad_results,
        rouge_results=rouge_results,
        bleu_results=bleu_results,
        mean_metrics=retrieval_metrics
    )

    results_path = os.path.join("results", "evaluation_results.json")
    print("\n✅ Saved. Verifying file:")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Total entries in file: {len(data)}")
        print("Last entry:")
        print(data[-1]["metrics"])
    else:
        print(f"Error: '{results_path}' not found.")

if __name__ == '__main__':
    main()
