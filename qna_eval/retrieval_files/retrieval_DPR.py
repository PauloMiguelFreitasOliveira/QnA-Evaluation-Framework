import os
import sys
import json
import numpy as np
import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, pipeline
import evaluate
import pytrec_eval

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

question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
model_name="ahotrod/electra_large_discriminator_squad2_512"
qa_pipeline = pipeline("question-answering", model=model_name , device=0 if torch.cuda.is_available() else -1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder.to(device)
context_encoder.to(device)

squad_metric = evaluate.load("squad")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

references, predictions, qrel, run = [], [], {}, {}

print_progress("Retrieving top-K contexts", 0.15)

for entry in query_passage_pairs:
    query = entry["query"]
    query_id = str(entry["query_id"])
    passages = entry["candidate_passages"]

    # DPR embeddings
    question_inputs = {k: v.to(device) for k, v in question_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).items()}
    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output.cpu().numpy()
    question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)

    context_embeddings = []
    for passage in passages:
        context_inputs = context_tokenizer(passage, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            emb = context_encoder(**context_inputs).pooler_output.squeeze().numpy()
        context_embeddings.append(emb)
    context_embeddings = np.array(context_embeddings)
    context_embeddings = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)

    # Similarity scores
    scores = np.dot(context_embeddings, question_embedding.squeeze())

    # Top passage
    best_idx = np.argmax(scores)
    top_context = passages[best_idx]
    result = qa_pipeline(question=query, context=top_context)
    predicted_answer = result["answer"]

    references.append({"id": query_id, "answers": {"text": entry["answers"], "answer_start": [0]}})
    predictions.append({"id": query_id, "prediction_text": predicted_answer})

    qrel[query_id] = {str(i): entry["is_selected"][i] for i in range(len(passages))}
    run[query_id] = {str(i): float(scores[i]) for i in range(len(passages))}

print_progress("Contexts retrieved", 0.30)
print_progress("Evaluating retrieval metrics", 0.50)

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'recip_rank'})
retrieval_metrics = evaluator.evaluate(run)
mean_metrics = {
    metric: np.mean([query_measures[metric] for query_measures in retrieval_metrics.values()])
    for metric in ['map', 'ndcg', 'recip_rank']
}

print_progress("Loading reader model", 0.65)
print_progress("Extracting answers with reader", 0.70)
print_progress("Evaluating QA metrics", 0.85)

squad_results = squad_metric.compute(predictions=predictions, references=references)
rouge_results = rouge_metric.compute(
    predictions=[pred["prediction_text"] for pred in predictions],
    references=[ref["answers"]["text"][0] for ref in references]
)
bleu_results = bleu_metric.compute(
    predictions=[pred["prediction_text"] for pred in predictions],
    references=[[ref["answers"]["text"][0]] for ref in references]
)

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
    "retrieval_method": "DPR",
    "evaluation_method": "Retrieval",
    "dataset_name": "ms_marco",
    "num_entries": len(query_passage_pairs),
    "metrics": {
        "retrieval": mean_metrics,
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
    mean_metrics=mean_metrics,
    examples=examples,
    num_entries=len(query_passage_pairs)
)

print_progress("Evaluation Complete", 1.0)
