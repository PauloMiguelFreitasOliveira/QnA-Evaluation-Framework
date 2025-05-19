import argparse
import os
import torch
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM,AutoConfig

from qna_eval.dataset_loader import load_dataset_file
from qna_eval.extractive_eval import evaluate_extractive_model
from logging_utils.save_results import save_evaluation_results

# Load .env into environment
load_dotenv()

# Now you can safely fetch your key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
client = OpenAI(api_key=OPENAI_API_KEY)


def is_seq2seq(model_name):
    """Determine if the model is encoder-decoder (seq2seq) or causal."""
    config = AutoConfig.from_pretrained(model_name)
    return config.is_encoder_decoder

def load_local_generator(model_name: str):
    """
    Load an HF pipeline for a local model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if is_seq2seq(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        task = "text-generation"
    return pipeline(
        task, model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

def generate_local_predictions(pipe, queries, prompt_template: str):
    preds = []
    for ex in tqdm(queries, desc="Generating (local)"):
        prompt = prompt_template.format(question=ex['query'])
        out = pipe(prompt)[0]
        ans = out.get("generated_text") or out.get("text") or ""
        preds.append({"query_id": ex['query_id'], "answer": ans.strip()})
    return preds


def generate_openai_predictions(model_name: str, queries, prompt_template: str):
    preds = []
    for ex in tqdm(queries, desc="Generating (OpenAI)" ):
        prompt = prompt_template.format(question=ex['query'])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128
        )
        ans = resp.choices[0].message.content.strip()
        preds.append({"query_id": ex['query_id'], "answer": ans})
    return preds


def build_ground_truth_dict(queries):
    return {ex["query_id"]: ex.get("answers", []) for ex in queries}


_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_truthful_metrics(predictions, ground_truth, threshold=0.2):
    """
    Hallucination = fraction of predictions with ZERO Rouge-L overlap vs all refs.
    Misinformation = fraction with Rouge-L < threshold.
    """
    total = len(predictions)
    if total == 0:
        return {"hallucination_rate": 0.0, "misinformation_rate": 0.0}

    no_overlap = 0
    low_overlap = 0

    for p in predictions:
        preds = p["answer"]
        refs = ground_truth.get(p["query_id"], [])

        # compute the maximum Rouge-L fmeasure against any reference
        best_f = 0.0
        for r in refs:
            score = _rouge_scorer.score(preds, r)["rougeL"].fmeasure
            if score > best_f:
                best_f = score

        if best_f == 0.0:
            no_overlap += 1
        if best_f < threshold:
            low_overlap += 1

    return {
        "hallucination_rate": round(no_overlap / total, 4),
        "misinformation_rate": round(low_overlap / total, 4)
    }


def load_secondary_dataset(name: str, limit: int):
    """
    Load a generative-only QA JSON (e.g. datasets/<name>.json).
    Only skips entries missing 'query'.
    """
    path = os.path.join("datasets", f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    formatted = []
    for i, ex in enumerate(raw):
        q = ex.get("query", "").strip()
        if not q:
            continue
        formatted.append({
            "query_id": ex.get("query_id", str(i)),
            "query": q,
            "answers": ex.get("answers", [])
        })
        if len(formatted) >= limit:
            break
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Run Generative QA + Hallucination Evaluation")
    parser.add_argument("--model_name", required=True,help="HuggingFace or OpenAI model name (e.g. flan-t5-base or gpt-4o)")
    parser.add_argument("--primary_dataset", required=True,help="Primary QA dataset (e.g. ms_marco)")
    parser.add_argument("--secondary_dataset", default="truthfulqa",help="Dataset for hallucination testing, default truthfulqa")
    parser.add_argument("--limit", type=int, default=500,help="Number of examples to evaluate per dataset")
    args = parser.parse_args()

    # Load primary dataset
    prim = load_dataset_file(args.primary_dataset, args.limit)
    prim_queries = prim['queries']

    # Generate answers for primary
    if args.model_name.startswith("gpt-") or args.model_name.startswith("gpt4"):
        prim_preds = generate_openai_predictions(args.model_name, prim_queries, "query: {question}")
    else:
        pipe = load_local_generator(args.model_name)
        prim_preds = generate_local_predictions(pipe, prim_queries, "query: {question}")

    # Evaluate primary QA
    prim_truth = build_ground_truth_dict(prim_queries)
    prim_metrics = evaluate_extractive_model(prim_preds, prim_truth)

    sec_q = load_secondary_dataset(args.secondary_dataset, args.limit)

    if args.model_name.startswith(("gpt-", "gpt4")):
        sec_preds = generate_openai_predictions(args.model_name, sec_q, "Question: {question}")
    else:
        sec_preds = generate_local_predictions(pipe, sec_q, "Question: {question}")

    sec_truth       = build_ground_truth_dict(sec_q)
    truth_metrics   = compute_truthful_metrics(sec_preds, sec_truth)

    # Save combined results
    save_evaluation_results(
        model_name=args.model_name,
        evaluation_method="Generative",
        dataset_name=args.primary_dataset,
        squad_results={
            "exact_match": prim_metrics['exact_match'],
            "f1": prim_metrics['f1']
        },
        rouge_results={k: prim_metrics[k] for k in ['rouge1','rouge2','rougeL','rougeLsum']},
        bleu_results={"bleu": prim_metrics['bleu'], "precisions": []},
        mean_metrics={},
        contextual_results=None,
        truth_metrics=truth_metrics,
        num_entries=len(prim_queries)
    )

if __name__ == '__main__':
    main()
