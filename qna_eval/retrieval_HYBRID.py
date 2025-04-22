if __name__ == '__main__':
    import os
    import sys
    import json
    import numpy as np
    import torch
    import faiss
    from tqdm import tqdm
    from rank_bm25 import BM25Okapi
    from transformers import (DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, AutoTokenizer, pipeline)
    import evaluate
    import pytrec_eval
    import random
    from nltk.tokenize import word_tokenize

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from logging_utils.save_results import save_evaluation_results

    # Load MS MARCO JSON file
    with open("..\\datasets\\ms_marco.json", "r", encoding="utf-8") as f:
        ms_marco_data = json.load(f)

    NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", 1000))
    ms_marco_data = ms_marco_data[:NUM_EXAMPLES]

    # Flatten structure
    query_passage_pairs = []
    for entry in ms_marco_data:
        query_passage_pairs.append({
            "query_id": entry["query_id"],
            "query": entry["query"],
            "answers": entry["answers"],
            "candidate_passages": [c["passage_text"] for c in entry["candidates"]],
            "is_selected": [c["is_selected"] for c in entry["candidates"]]
        })

    # Tokenizers and models
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

    all_contexts = list({ctx for pair in query_passage_pairs for ctx in pair["candidate_passages"]})

    # Build DPR context embeddings and FAISS index
    BATCH_SIZE = 16
    context_embeddings = []

    for i in tqdm(range(0, len(all_contexts), BATCH_SIZE), desc="Encoding contexts in batches"):
        batch = all_contexts[i:i + BATCH_SIZE]
        inputs = context_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = context_encoder(**inputs).pooler_output
        context_embeddings.extend(outputs.cpu().numpy())



    context_embeddings = np.array(context_embeddings).astype("float32")
    faiss.normalize_L2(context_embeddings)
    faiss_index = faiss.IndexFlatIP(context_embeddings.shape[1])
    faiss_index.add(context_embeddings)

    # BM25 indexing
    #tokenized_contexts = [bert_tokenizer.tokenize(context) for context in all_contexts]
    tokenized_contexts = [word_tokenize(context.lower()) for context in all_contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    model_name = "ahotrod/electra_large_discriminator_squad2_512"

    # QA model
    qa_pipeline = pipeline("question-answering", model=model_name)

    # Evaluation metrics
    squad_metric = evaluate.load("squad")
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    references, predictions, qrel, run = [], [], {}, {}

    def hybrid_retrieval(question, top_n=10, alpha=0.5):
        # DPR retrieval
        inputs = question_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            question_emb = question_encoder(**inputs).pooler_output.squeeze().numpy()
        faiss.normalize_L2(question_emb.reshape(1, -1))
        dpr_scores, dpr_indices = faiss_index.search(np.array([question_emb]), top_n)
        dpr_contexts = [all_contexts[idx] for idx in dpr_indices[0]]

        # BM25 retrieval
        tokenized_question = word_tokenize(question.lower())
        #tokenized_question = bert_tokenizer.tokenize(question)
        bm25_scores = bm25.get_scores(tokenized_question)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_n]
        bm25_contexts = [all_contexts[idx] for idx in bm25_indices]

        # Fusion
        combined = list(set(dpr_contexts + bm25_contexts))
        fusion_scores = {}
        for ctx in combined:
            dpr_score = dpr_scores[0][dpr_contexts.index(ctx)] if ctx in dpr_contexts else 0
            bm25_score = bm25_scores[all_contexts.index(ctx)] if ctx in bm25_contexts else 0

            # Normalize
            if len(dpr_scores[0]) > 1:
                dpr_score = (dpr_score - np.min(dpr_scores[0])) / (np.max(dpr_scores[0]) - np.min(dpr_scores[0]))
            if len(bm25_scores) > 1:
                bm25_score = (bm25_score - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

            fusion_scores[ctx] = alpha * dpr_score + (1 - alpha) * bm25_score

        sorted_fusion = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        top_contexts = [x[0] for x in sorted_fusion[:top_n]]
        top_scores = [x[1] for x in sorted_fusion[:top_n]]
            # 🧪 Optional: Debug print
        if random.random() < 0.01:  # or `if True:` to show every query
            print("\n" + "="*100)
            print(f"🔍 Query ID: {query_id}")
            print(f"📌 Question: {question}")
            if true_answers:
                print(f"🎯 Ground Truth: {true_answers}")
            print("="*100)

            # Get original rank positions
            dpr_rankings = np.argsort(dpr_scores[0])[::-1]
            bm25_rankings = np.argsort(bm25_scores)[::-1]

            dpr_rank_map = {all_contexts[idx]: rank for rank, idx in enumerate(dpr_indices[0])}
            bm25_rank_map = {all_contexts[idx]: rank for rank, idx in enumerate(bm25_indices)}

            print(f"\n📊 Top {top_n} Fusion Passages:")
            print("-" * 100)
            print(f"{'FusionRank':>10} | {'FusionScore':>12} | {'DPR_Rank':>9} | {'BM25_Rank':>10} | Passage")
            print("-" * 100)

            for fusion_rank, ctx in enumerate(top_contexts):
                dpr_rank = dpr_rank_map.get(ctx, "-")
                bm25_rank = bm25_rank_map.get(ctx, "-")
                print(f"{fusion_rank+1:>10} | {top_scores[fusion_rank]:>12.4f} | {str(dpr_rank):>9} | {str(bm25_rank):>10} | {ctx[:80]}...")

            print("=" * 100 + "\n")

        return top_contexts, top_scores

    # Evaluation loop
    for pair in tqdm(query_passage_pairs, desc="Evaluating Hybrid Retrieval"):
        query_id = pair["query_id"]
        query = pair["query"]
        true_answers = pair["answers"]

        retrieved_contexts, scores = hybrid_retrieval(query, top_n=10)
        combined_context = " ".join(retrieved_contexts[:5])
        result = qa_pipeline(question=query, context=combined_context)
        pred_answer = result["answer"]

        references.append({
            "id": query_id,
            "answers": {
                "text": true_answers,
                "answer_start": [0] * len(true_answers)  
            }
        })
        predictions.append({"id": query_id, "prediction_text": pred_answer})

        qrel[query_id] = {}
        for i, ctx in enumerate(retrieved_contexts):
            if ctx in pair["candidate_passages"]:
                idx = pair["candidate_passages"].index(ctx)
                qrel[query_id][str(i)] = int(pair["is_selected"][idx])
            else:
                qrel[query_id][str(i)] = 0

        run[query_id] = {str(i): float(scores[i]) for i in range(len(scores))}

    # Metric calculations
    squad_results = squad_metric.compute(predictions=predictions, references=references)
    rouge_results = rouge_metric.compute(predictions=[p["prediction_text"] for p in predictions],
                                        references=[r["answers"]["text"][0] for r in references])
    bleu_results = bleu_metric.compute(predictions=[p["prediction_text"] for p in predictions],
                                    references=[[r["answers"]["text"][0]] for r in references])

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"map", "ndcg", "recip_rank"})
    retrieval_metrics = evaluator.evaluate(run)

    mean_retrieval_metrics = {metric: np.mean([m[metric] for m in retrieval_metrics.values()])
                            for metric in ["map", "ndcg", "recip_rank"]}

    # Print and save results
    print(f"Exact Match: {squad_results['exact_match']:.2f}")
    print(f"F1 Score: {squad_results['f1']}")
    print(f"ROUGE-l F1: {rouge_results['rouge1']}")
    print(f"BLEU Score: {bleu_results['bleu']}")
    print(f"MAP: {mean_retrieval_metrics['map']}")
    print(f"NDCG: {mean_retrieval_metrics['ndcg']}")
    print(f"MRR: {mean_retrieval_metrics['recip_rank']}")

    save_evaluation_results(
        model_name=model_name,
        retrieval_method="Hybrid",
        dataset_name="ms_marco",
        squad_results=squad_results,
        rouge_results=rouge_results,
        bleu_results=bleu_results,
        mean_metrics=mean_retrieval_metrics
    )
