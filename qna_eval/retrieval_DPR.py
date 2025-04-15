if __name__ == '__main__':
    from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, pipeline
    import evaluate
    import pytrec_eval
    import torch
    import numpy as np
    import json
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from logging_utils.save_results import save_evaluation_results

    # Load MS MARCO JSON file
    with open("..\\datasets\\ms_marco.json", "r", encoding="utf-8") as f:
        ms_marco_data = json.load(f)

    NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", 1000))  # Adjust as needed
    ms_marco_data = ms_marco_data[:NUM_EXAMPLES]

    # Flatten entries into query-passage structure
    query_passage_pairs = []
    for entry in ms_marco_data:
        query_passage_pairs.append({
            "query_id": entry["query_id"],
            "query": entry["query"],
            "answers": entry["answers"],
            "candidate_passages": [c["passage_text"] for c in entry["candidates"]],
            "is_selected": [c["is_selected"] for c in entry["candidates"]]
        })

    # Load DPR model/tokenizers
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

    model_name="ahotrod/electra_large_discriminator_squad2_512"

    # QA pipeline (same as BM25 baseline)
    qa_pipeline = pipeline("question-answering", model=model_name , device=0 if torch.cuda.is_available() else -1)

    # Evaluation metrics
    squad_metric = evaluate.load("squad")
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    references, predictions = [], []
    qrel, run = {}, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_encoder.to(device)
    context_encoder.to(device)

    # Evaluation loop
    for entry in query_passage_pairs:
        query = entry["query"]
        query_id = str(entry["query_id"])
        passages = entry["candidate_passages"]
        true_answer = entry["answers"][0] if entry["answers"] else ""

        # DPR question embedding
        question_inputs = {k: v.to(device) for k, v in question_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).items()}

        with torch.no_grad():
            question_embedding = question_encoder(**question_inputs).pooler_output.cpu().numpy()

        question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)

        # DPR context embeddings
        context_embeddings = []
        for passage in passages:
            context_inputs = context_tokenizer(passage, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                emb = context_encoder(**context_inputs).pooler_output.squeeze().numpy()
            context_embeddings.append(emb)

        context_embeddings = np.array(context_embeddings)
        context_embeddings = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)

        # Similarity scores via dot product (cosine)
        scores = np.dot(context_embeddings, question_embedding.squeeze())

        # Pick best passage
        top_k = 3  # you can try 5 as well
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        top_contexts = [passages[i] for i in top_k_indices]
        combined_context = " ".join(top_contexts)
        result = qa_pipeline(question=query, context=combined_context)

        predicted_answer = result["answer"]

        # Append for metrics
        references.append({"id": query_id, "answers": {"text": entry["answers"], "answer_start": [0]}})
        predictions.append({"id": query_id, "prediction_text": predicted_answer})

        # Prepare for pytrec_eval
        qrel[query_id] = {str(i): entry["is_selected"][i] for i in range(len(passages))}
        run[query_id] = {str(i): float(scores[i]) for i in range(len(passages))}

        



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
    print(f"Exact Match: {squad_results['exact_match']:.2f}")
    print(f"F1 Score: {squad_results['f1']:.2f}")
    print(f"ROUGE-l F1 Score: {rouge_results['rouge1']:.4f}")
    print(f"BLEU Score: {bleu_results['bleu']:.4f}")

    # Retrieval metrics
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'recip_rank'})
    retrieval_metrics = evaluator.evaluate(run)

    mean_metrics = {
        metric: np.mean([query_measures[metric] for query_measures in retrieval_metrics.values()])
        for metric in ['map', 'ndcg', 'recip_rank']
    }

    print(f"\n📊 DPR Ranking Evaluation")
    print(f"Mean Average Precision (MAP):     {mean_metrics['map']:.4f}")
    print(f"Normalized DCG (NDCG):            {mean_metrics['ndcg']:.4f}")
    print(f"Mean Reciprocal Rank (MRR):       {mean_metrics['recip_rank']:.4f}")

    # Save results
    save_evaluation_results(
        model_name=model_name,
        retrieval_method="DPR",
        dataset_name="ms_marco",
        squad_results=squad_results,
        rouge_results=rouge_results,
        bleu_results=bleu_results,
        mean_metrics=mean_metrics
    )
