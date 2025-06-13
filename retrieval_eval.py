import argparse
from qna_eval.retriever import retrieve_top_k
from qna_eval.reader import load_reader, extract_answers
from qna_eval.dataset_loader import load_dataset_file
from qna_eval.evaluator import evaluate_retrieval, evaluate_qa, evaluate_contextual
from logging_utils.save_results import save_evaluation_results
from qna_eval.retriever_bm25 import retrieve_top_k_bm25
import sys
import json

def print_progress(stage, progress):
    sys.stdout.flush()
    print(json.dumps({"process_stage": stage, "progress": progress}))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Retriever and Reader Models on QnA Datasets")
    parser.add_argument('--retriever_model', required=True, help='Retriever model name or path')
    parser.add_argument('--reader_model', default=None, help='Reader (extractive QA) model name')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--top_k', type=int, default=10, help='Top K passages to retrieve')
    parser.add_argument('--limit', type=int, default=1000, help='Limit number of queries for testing')
    parser.add_argument('--max_context_per_query', type=int, default=100, help='Limit number of contextes per query')
    args = parser.parse_args()

    # Progress 1: Loading dataset
    print_progress("Loading dataset", 0.05)
    dataset = load_dataset_file(args.dataset, args.limit, args.max_context_per_query)
    queries = dataset["queries"]
    context_pool = dataset.get("context_pool")

    # Progress 2: Retrieving contexts
    print_progress("Retrieving top-K contexts", 0.15)
    if args.retriever_model.lower() == "bm25":
        results = retrieve_top_k_bm25(queries, top_k=args.top_k, context_pool=context_pool)
    else:
        results = retrieve_top_k(args.retriever_model, queries, top_k=args.top_k, context_pool=context_pool)
    print_progress("Contexts retrieved", 0.30)

    # Prepare retrieval metrics inputs
    qrel = {r["query_id"]: r["qrel"] for r in results}
    run  = {r["query_id"]: r["run"]  for r in results}

    # Progress 3: Evaluate retrieval
    print_progress("Evaluating retrieval metrics", 0.50)
    retrieval_metrics = evaluate_retrieval(qrel, run)

    # QA (reader) metrics
    qa_metrics = {}
    contextual_metrics = {}
    if args.reader_model:
        print_progress("Loading reader model", 0.65)
        reader = load_reader(args.reader_model)
        print_progress("Extracting answers with reader", 0.70)
        predictions = extract_answers(reader, results, separator=" [SEP] ")
        print_progress("Evaluating QA metrics", 0.85)
        qa_metrics = evaluate_qa(queries, predictions)
        contextual_metrics = evaluate_contextual(results)

    # Prepare metrics for frontend and save_results.py
    print_progress("Saving results", 0.98)
    squad_results = {
        "exact_match": qa_metrics.get("exact_match", 0.0),
        "f1": qa_metrics.get("f1", 0.0),
        "rouge1": qa_metrics.get("rouge1", 0.0),
        "rouge2": qa_metrics.get("rouge2", 0.0),
        "rougeL": qa_metrics.get("rougeL", 0.0),
        "rougeLsum": qa_metrics.get("rougeLsum", 0.0),
        "bleu": qa_metrics.get("bleu", 0.0)
    }
    rouge_results = {k: squad_results[k] for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    bleu_results = {"bleu": qa_metrics.get("bleu", 0.0), "precisions": []}

    # Optional: Include 5 example Q&A pairs for the UI
    examples = []
    if args.reader_model and results and predictions:
        for i in range(min(5, len(results))):
            ex = {
                "query_id": results[i]["query_id"],
                "query": results[i].get("query"),
                "ground_truth": results[i].get("answers", []),
                "prediction": predictions[i].get("answer", "") if i < len(predictions) else ""
            }
            examples.append(ex)

    # FINAL OUTPUT: Only one line, like generative/rag!
    output_json = {
        "model_name": args.reader_model or args.retriever_model,
        "retrieval_method": args.retriever_model,
        "evaluation_method": "Retrieval",
        "dataset_name": args.dataset,
        "num_entries": len(queries),
        "metrics": {
            "retrieval": retrieval_metrics,
            "squad": squad_results,
            "rouge": rouge_results,
            "bleu": bleu_results,
            "contextual": contextual_metrics
        },
        "examples": examples
    }
    print(json.dumps(output_json, ensure_ascii=False))
    save_evaluation_results(
        model_name=args.reader_model or args.retriever_model,
        evaluation_method="Retrieval",
        dataset_name=args.dataset,
        squad_results=squad_results,
        rouge_results=rouge_results,
        bleu_results=bleu_results,
        mean_metrics=retrieval_metrics,
        contextual_results=contextual_metrics,
        num_entries=len(queries),
        examples=examples
    )
    print_progress("Evaluation Complete", 1.0)

if __name__ == '__main__':
    main()
