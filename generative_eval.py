import argparse
import os
import json
import random
import subprocess
import torch
import sys
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

from qna_eval.dataset_loader import load_dataset_file
from qna_eval.extractive_eval import evaluate_extractive_model
from logging_utils.save_results import save_evaluation_results
from qna_eval.hallucination_eval import load_secondary_dataset, generate_multisample, judge_with_llm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── CONFIG ──────────────────────────────────────────────────────────────────────
JUDGE_MODEL = "gpt-4o-mini"
OLLAMA_CMD = "ollama"
# ────────────────────────────────────────────────────────────────────────────────

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
client = OpenAI(api_key=OPENAI_API_KEY)

def is_seq2seq(model_name):
    cfg = AutoConfig.from_pretrained(model_name)
    return cfg.is_encoder_decoder

def load_local_generator(model_name: str):
    """Loads tokenizer, model and chooses pipeline task"""
    tok = AutoTokenizer.from_pretrained(model_name)
    if is_seq2seq(model_name):
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        task = "text2text-generation"
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        task = "text-generation"
    return pipeline(
        task,
        model=mdl,
        tokenizer=tok,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tok.eos_token_id
    )

def generate_openai_multisample(model_name, queries, prompt_tpl, n_samples, temp):
    """One-shot n-sample generation via OpenAI’s `n` parameter."""
    outs = []
    for ex in queries:
        prompt = prompt_tpl.format(question=ex["query"])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=temp,
            max_tokens=128,
            n=n_samples
        )
        for choice in resp.choices:
            outs.append({
                "query_id": ex["query_id"],
                "answer": choice.message.content.strip()
            })
    return outs


def generate_ollama_predictions(
    model_name: str,
    queries: list[dict],
    prompt_tpl: str,
    temperature: float = 0.0
) -> list[dict]:
    """
    For each query, call:
      ollama run <model_name> "<prompt>"
    Capture raw bytes and decode with utf-8 (replacing invalid sequences).
    """
    preds = []
    for ex in queries:
        prompt = prompt_tpl.format(question=ex["query"]).strip()
        cmd = [
            OLLAMA_CMD,
            "run", model_name,
            prompt
        ]
        try:
            # capture raw bytes to avoid CP1252 decode errors
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                check=True
            )
        except subprocess.CalledProcessError as e:
            # print the failing prompt and stderr (decoded safely)
            print(f"\n––– Ollama CLI failed for prompt:\n{prompt}\n")
            print("---- stderr start ----")
            err_raw = e.stderr or b""
            err_text = err_raw.decode("utf-8", errors="replace")
            print(err_text.strip() or "<no stderr>")
            print("---- stderr end  ----\n")
            raise

        # decode the model’s output
        raw = proc.stdout or b""
        ans = raw.decode("utf-8", errors="replace").strip()

        preds.append({
            "query_id": ex["query_id"],
            "answer": ans
        })

    return preds


def build_ground_truth(queries):
    """returns a dictionary to be used in evaluate_extractive_model"""
    return {ex["query_id"]: ex.get("answers", []) for ex in queries}

def print_progress(stage, progress):
    sys.stdout.flush()
    print(json.dumps({"process_stage": stage, "progress": progress}))
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",       required=True)
    p.add_argument("--primary_dataset",  required=True)
    p.add_argument("--secondary_dataset",default="truthfulqa")
    p.add_argument("--limit",  type=int, default=500)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--backend", choices=["openai","ollama","hf"], default="openai")
    args = p.parse_args()

    prompt_tpl = "Question: {question}"

    # ── Step 1: Primary QA ────────────────────────────────────────────────────────
    ds       = load_dataset_file(args.primary_dataset, args.limit)
    prim_qs  = ds["queries"]

    print_progress("Generating primary answers", 0.15)
    if args.backend == "openai":
        prim_preds = generate_openai_multisample(args.model_name, prim_qs, prompt_tpl, 1, temp=0.0)
    elif args.backend == "ollama":
        prim_preds = generate_ollama_predictions(args.model_name, prim_qs, prompt_tpl)
    else:
        local_pipe = load_local_generator(args.model_name)
        prim_preds = [
            {"query_id": ex["query_id"],
             "answer": local_pipe(prompt_tpl.format(question=ex["query"]))[0].get("generated_text","").strip()}
            for ex in (prim_qs)]
    print_progress("Evaluating QA metrics", 0.25)
    prim_truth   = build_ground_truth(prim_qs)
    prim_metrics = evaluate_extractive_model(prim_preds, prim_truth)

    # ── Step 2: Hallucination via fixed judge ────────────────────────────────────
    print_progress("Loading secondary dataset", 0.30)
    sec_limit = max(1, args.limit // args.n_samples)
    sec_qs    = load_secondary_dataset(args.secondary_dataset, sec_limit)

    print_progress("Generating secondary predictions (hallucination eval)", 0.35)
    if args.backend == "openai":
        sec_preds = generate_openai_multisample(
            args.model_name, sec_qs, prompt_tpl, args.n_samples, temp=0.7
        )
    elif args.backend == "ollama":
        sec_preds = []
        for _ in range(args.n_samples):
            sec_preds += generate_ollama_predictions(args.model_name, sec_qs, prompt_tpl)
    else:
        local_pipe = load_local_generator(args.model_name)
        sec_preds = generate_multisample(
            local_pipe, sec_qs, prompt_tpl,
            n_samples=args.n_samples,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128
        )

    # always use the *fixed* judge  
    print_progress("Judging hallucinations", 0.50)
    judgement = judge_with_llm(
        client,
        JUDGE_MODEL,
        sec_qs,
        sec_preds,
        fs_examples=None
    )

    # pick 5 random examples to save/display
    print_progress("Preparing examples and output", 0.80)
    idxs = random.sample(range(len(prim_qs)), k=min(5,len(prim_qs)))
    examples = [{
        "query_id":   prim_qs[i]["query_id"],
        "query":      prim_qs[i]["query"],
        "ground_truth": prim_qs[i]["answers"],
        "prediction": prim_preds[i]["answer"]
    } for i in idxs]

    out = {
        "model_name":        args.model_name,
        "evaluation_method": "Generative",
        "primary_dataset":   args.primary_dataset,
        "secondary_dataset": args.secondary_dataset,
        "num_primary":       len(prim_qs),
        "num_secondary":     len(sec_qs),
        "metrics": {
            "squad":         prim_metrics,
            "hallucination": judgement
        },
        "examples": examples
    }
    print_progress("Saving results", 0.90)
    print(json.dumps(out, ensure_ascii=False))


    save_evaluation_results(
        model_name=args.model_name,
        evaluation_method="Generative",
        dataset_name=args.primary_dataset,
        squad_results = {k: prim_metrics.get(k, 0) for k in ["exact_match", "f1"]},
        rouge_results = {k: prim_metrics.get(k, 0) for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]},
        bleu_results  = {"bleu": prim_metrics.get("bleu", 0), "precisions":[]},
        mean_metrics={},
        contextual_results=None,
        truth_metrics=judgement,
        examples=examples,
        num_entries=len(prim_qs)
    )
    print_progress("Evaluation Complete", 1.0)

if __name__ == "__main__":
    main()
