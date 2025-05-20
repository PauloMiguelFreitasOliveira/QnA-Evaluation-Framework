import argparse
import os
import torch
import json
from tqdm import tqdm
import random
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM,AutoConfig

from qna_eval.dataset_loader import load_dataset_file
from qna_eval.extractive_eval import evaluate_extractive_model
from logging_utils.save_results import save_evaluation_results
from qna_eval.hallucination_eval import (load_secondary_dataset, generate_multisample, judge_with_llm)

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


def generate_openai_predictions(model_name: str, queries, prompt_tpl: str, temperature: float):
    preds = []
    for ex in tqdm(queries, desc="Generating (OpenAI)"):
        prompt = prompt_tpl.format(question=ex["query"])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=128
        )
        ans = resp.choices[0].message.content.strip()
        preds.append({"query_id": ex["query_id"], "answer": ans})
    return preds

def build_ground_truth(queries):
    return {ex["query_id"]: ex.get("answers", []) for ex in queries}


def main():
    parser = argparse.ArgumentParser(description="Run Generative QA + Hallucination Evaluation")
    parser.add_argument("--model_name", required=True,help="HuggingFace or OpenAI model name (e.g. flan-t5-base or gpt-4o)")
    parser.add_argument("--primary_dataset", required=True,help="Primary QA dataset (e.g. ms_marco)")
    parser.add_argument("--secondary_dataset", default="truthfulqa",help="Dataset for hallucination testing, default truthfulqa")
    parser.add_argument("--limit", type=int, default=500,help="Number of examples to evaluate per dataset")
    parser.add_argument("--n_samples", type=int, default=10, help="How many samples per query for hallucination")
    args = parser.parse_args()

    # decide whether to call OpenAI vs local
    use_openai = args.model_name.startswith(("gpt-", "gpt4"))
    prompt_tpl = "Question: {question}"


    # 1) Primary QA evaluation
    ds = load_dataset_file(args.primary_dataset, args.limit)
    prim_qs = ds["queries"]
    if use_openai:
        prim_preds = generate_openai_predictions(args.model_name, prim_qs, prompt_tpl, temperature=0.0)
    else:
        local_pipe = load_local_generator(args.model_name)
        prim_preds = generate_local_predictions(local_pipe, prim_qs, prompt_tpl)

    prim_truth   = build_ground_truth(prim_qs)
    prim_metrics = evaluate_extractive_model(prim_preds, prim_truth)


    # 2) Hallucination on secondary dataset
    # decide how many queries for multi‐sampling so that limit queries × n_samples ~= args.limit
    sec_limit = max(1, args.limit // args.n_samples)
    sec_qs = load_secondary_dataset(args.secondary_dataset, sec_limit)

    if use_openai:
        # sample with temperature >0 to get diversity
        sec_preds = generate_openai_predictions(args.model_name, sec_qs, prompt_tpl, temperature=0.7) * args.n_samples
    else:
        local_pipe = load_local_generator(args.model_name)
        sec_preds = generate_multisample(
            local_pipe,
            sec_qs,
            prompt_tpl,
            n_samples=args.n_samples,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128
        )

    # now have sec_qs (list of queries) and sec_preds (flat list of samples)
    judgement = judge_with_llm(
        client,
        args.model_name,
        sec_qs,
        sec_preds,
        fs_examples=None  # or you could build few-shot examples here
    )

    sample_idxs = random.sample(range(len(prim_qs)), k=min(5, len(prim_qs)))
    examples = []
    for i in sample_idxs:
        q = prim_qs[i]
        p = prim_preds[i]
        examples.append({
            "query_id": q["query_id"],
            "query":    q["query"],
            "ground_truth": q["answers"],
            "prediction":   p["answer"]
        })

    # ─── Report & Save ────────────────────────────────────────────────────────────
    out = {
        "model_name": args.model_name,
        "evaluation_method": "Generative",
        "primary_dataset": args.primary_dataset,
        "secondary_dataset": args.secondary_dataset,
        "num_primary": len(prim_qs),
        "num_secondary": len(sec_qs),
        "metrics": {
            "squad":     prim_metrics,
            "hallucination": judgement
        }
    }
    print(json.dumps(out, indent=2))

    save_evaluation_results(
        model_name=args.model_name,
        evaluation_method="Generative",
        dataset_name=args.primary_dataset,
        squad_results={ "exact_match": prim_metrics["exact_match"],
                        "f1":          prim_metrics["f1"] },
        rouge_results={ k: prim_metrics[k]
                        for k in ["rouge1","rouge2","rougeL","rougeLsum"] },
        bleu_results={ "bleu": prim_metrics["bleu"], "precisions": [] },
        mean_metrics={},
        contextual_results=None,
        truth_metrics=judgement,
        examples=examples, 
        num_entries=len(prim_qs)
    )

if __name__ == "__main__":
    main()