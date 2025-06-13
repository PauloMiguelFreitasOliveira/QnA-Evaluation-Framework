import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import (DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, AutoTokenizer, pipeline)
import evaluate
import pytrec_eval
import random
from nltk.tokenize import word_tokenize

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
from logging_utils.save_results import save_evaluation_results

def print_progress(stage, progress):
    sys.stdout.flush()
    print(json.dumps({"process_stage": stage, "progress": progress}))
    sys.stdout.flush()

dataset_path = os.path.join(ROOT_DIR, "datasets", "ms_marco.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    ms_marco_data = json.load(f)

NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", 100))
ms_marco_data = ms_marco_data[:NUM_EXAMPLES]

print_progress("Loading dataset", 0.05)

query_passage_pairs = []
for entry in ms_marco_data:
    query_passage_pairs.append({
        "query_id": entry["query_id"],
        "query": entry["query"],
        "answers": entry["answers"],
        "candidate_passages": [c["passage_text"] for c in entry["candidates"]],
        "is_selected": [c["is_selected"] for c in entry["candidates"]]
    })

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

all_contexts = list({ctx for pair in query_passage_pairs for ctx in pair["candidate_passages"]})

# Build context embeddings
BATCH_SIZE = 16
context_embeddings = []
for i in tqdm(range(0, len(all_contexts), BATCH_SIZE), desc="Encoding contexts in batches"):
    batch = all_contexts[i:i + BATCH_SIZE]
    inputs = context_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = context_encoder(**inputs).pooler_output
    context_embeddings.extend(outputs.cpu().numpy())

context_embeddings = np.array(context_embeddings).astype("float32")
import faiss
faiss.normalize_L2(context_embeddings)
faiss_index = faiss.IndexFlatIP(context_embeddings.shape[1])
faiss_index.add(context_embeddings)

tokenized_contexts = [word_tokenize(context.lower()) for context in all_contexts]
bm25 = BM25Okapi(tokenized_contexts)
model_name = "ahotrod/electra_large_discriminator_squad2_512"
qa_pipeline = pipeline("question-answering", model=model_name)

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
    bm25_scores = bm25.get_scores(tokenized_question)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_n]
    bm25_contexts = [all_contexts[idx] for idx in bm25_indices]

    # Fusion
    combined = list(set(dpr_contexts + bm25_contexts))
    fusion_scores = {}
    for ctx in combined:
        dpr_score = dpr_scores[0][dpr_contexts.index(ctx)] if ctx in dpr_contexts else 0
        bm25_score = bm25_scores[all_contexts.index(ctx)] if ctx in bm25_contexts else 0
        if len(dpr_scores[0]) > 1:
            dpr_score = (dpr_score - np.min(dpr_scores[0])) / (np.max(dpr_scores[0]) - np.min(dpr_scores[0]))
        if len(bm25_scores) > 1:
            bm25_score = (bm25_score - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        fusion_scores[ctx] = alpha * dpr_score + (1 - alpha) * bm25_score

    sorted_fusion = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    top_contexts = [x[0] for x in sorted_fusion[:top_n]]
    top_scores = [x[1] for x in sorted_fusion[:top_n]]
    return top_contexts, top_scores

print_progress("Retrieving top-K contexts", 0.15)

for pair in tqdm(query_passage_pairs, desc="Evaluating Hybrid Retrieval"):
    query_id = pair["query_id"]
    query = pair["query"]
    true_answers = pair["answers"]

    retrieved_contexts, scores = hybrid_retrieval(query, top_n=10)
    top_context = retrieved_contexts[0]
    result = qa_pipeline(question=query, context=top_context)
    pred_answer = result["answer"]

    references.append({
        "id": query_id,
        "answers": {"text": true_answers, "answer_start": [0] * len(true_answers)}
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

print_progress("Contexts retrieved", 0.30)
print_progress("Evaluating retrieval metrics", 0.50)

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"map", "ndcg", "recip_rank"})
retrieval_metrics = evaluator.evaluate(run)
mean_retrieval_metrics = {metric: np.mean([m[metric] for m in retrieval_metrics.values()])
                        for metric in ["map", "ndcg", "recip_rank"]}

print_progress("Loading reader model", 0.65)
print_progress("Extracting answers with reader", 0.70)
print_progress("Evaluating QA metrics", 0.85)

squad_results = squad_metric.compute(predictions=predictions, references=references)
rouge_results = rouge_metric.compute(predictions=[p["prediction_text"] for p in predictions],
                                    references=[r["answers"]["text"][0] for r in references])
bleu_results = bleu_metric.compute(predictions=[p["prediction_text"] for p in predictions],
                                references=[[r["answers"]["text"][0]] for r in references])

print_progress("Saving results", 0.98)

examples = []
for i in range(min(5, len(query_passage_pairs))):
    ex = query_passage_pairs[i]
    examples.append({
        "query_id": ex["query_id"],
        "query": ex["query"],
        "ground_truth": ex["answers"],
        "prediction": predictions[i]["prediction_text"]
    })

output_json = {
    "model_name": model_name,
    "retrieval_method": "Hybrid",
    "evaluation_method": "Retrieval",
    "dataset_name": "ms_marco",
    "num_entries": len(query_passage_pairs),
    "metrics": {
        "retrieval": mean_retrieval_metrics,
        "squad": squad_results,
        "rouge": rouge_results,
        "bleu": bleu_results,
        "contextual": {}
    },
    "examples": examples
}
print(json.dumps(output_json, ensure_ascii=False))

save_evaluation_results(
    model_name=model_name,
    evaluation_method="Retrieval",
    dataset_name="ms_marco",
    squad_results=squad_results,
    rouge_results=rouge_results,
    bleu_results=bleu_results,
    mean_metrics=mean_retrieval_metrics,
    examples=examples,
    num_entries=len(query_passage_pairs)
)

print_progress("Evaluation Complete", 1.0)
