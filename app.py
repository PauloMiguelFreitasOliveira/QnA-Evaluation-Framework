import argparse
from retriever import load_retriever, retrieve_top_k
from reader import load_reader, extract_answers
from dataset_loader import load_dataset
from evaluator import evaluate_retrieval, evaluate_qa
from logger import save_evaluation_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_model', required=True, help='Retriever model name or path')
    parser.add_argument('--reader_model', default=None, help='Reader (extractive QA) model name')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--top_k', type=int, default=10, help='Top K passages to retrieve')
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Load retriever
    retriever = load_retriever(args.retriever_model)

    # Get top-K retrievals
    retrieved_passages = retrieve_top_k(retriever, dataset, top_k=args.top_k)

    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(dataset, retrieved_passages)

    if args.reader_model:
        # Load reader
        reader = load_reader(args.reader_model)

        # Extract answers
        predictions = extract_answers(reader, retrieved_passages)

        # Evaluate QA
        qa_metrics = evaluate_qa(dataset, predictions)
    else:
        qa_metrics = {}

    # Save everything
    save_evaluation_results(
        model_name=args.retriever_model,
        reader_model=args.reader_model,
        dataset_name=args.dataset,
        retrieval_metrics=retrieval_metrics,
        qa_metrics=qa_metrics,
    )

if __name__ == '__main__':
    main()
