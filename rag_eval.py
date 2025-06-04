import argparse
import os
import random
import json
import subprocess
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

from qna_eval.dataset_loader import load_dataset_file
from qna_eval.retriever import retrieve_top_k
from qna_eval.extractive_eval import evaluate_extractive_model
from logging_utils.save_results import save_evaluation_results

# Fixed retriever model for ranking documents
RETRIEVER_MODEL = "facebook/dpr-question_encoder-single-nq-base"
JUDGE_MODEL = "gpt-4o-mini"
OLLAMA_CMD = "ollama"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
client = OpenAI(api_key=OPENAI_API_KEY)


def is_seq2seq(model_name):
    cfg = AutoConfig.from_pretrained(model_name)
    return cfg.is_encoder_decoder


def load_local_generator(model_name: str):
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


def generate_openai(model_name: str, prompts: list[tuple[str, str]], n_samples: int = 1, temp: float = 0.0):
    outs = []
    for qid, prompt in tqdm(prompts, desc="Generating (OpenAI)"):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=128,
            n=n_samples
        )
        for choice in resp.choices:
            outs.append({"query_id": qid, "answer": choice.message.content.strip()})
    return outs


def generate_ollama(model_name: str, prompts: list[tuple[str, str]]):
    preds = []
    for qid, prompt in tqdm(prompts, desc=f"Generating (Ollama:{model_name})"):
        cmd = [OLLAMA_CMD, "run", model_name, prompt]
        proc = subprocess.run(cmd, capture_output=True, text=False, check=True)
        ans = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        preds.append({"query_id": qid, "answer": ans})
    return preds


def build_prompt(contexts: list[str], question: str) -> str:
    context_block = "\n".join(contexts)
    return f"{context_block}\n\nQuestion: {question}\nAnswer:"


def build_ground_truth(queries):
    return {ex["query_id"]: ex.get("answers", []) for ex in queries}


def judge_with_context(client: OpenAI, model_name: str, retrieved: list[dict], predictions: list[dict]):
    by_qid = {}
    for p in predictions:
        by_qid.setdefault(p["query_id"], []).append(p["answer"])

    total_q = len(retrieved)
    phr = thr = 0

    for item in tqdm(retrieved, desc="Judging hallucinations"):
        qid = item["query_id"]
        question = item["query"]
        context_block = "\n".join(item["top_passages"])
        answers = by_qid.get(qid, [])

        flags = []
        for ans in answers:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checker. Given a question, context and a candidate answer, "
                        "reply exactly SUPPORT if the answer is fully supported by the context, "
                        "or HALLUCINATION otherwise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nContext:\n{context_block}\n\n"
                        f"Candidate answer:\n{ans}\n\nIs the answer fully supported by the context?"
                    ),
                },
            ]
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=3,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            flags.append(verdict.startswith("SUPPORT"))

        if any(not f for f in flags):
            phr += 1
        if all(not f for f in flags):
            thr += 1

    return {"PHR": round(phr / total_q, 4), "THR": round(thr / total_q, 4)}


def main():
    p = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation evaluation")
    p.add_argument("--model_name", required=True, help="Generative model to use")
    p.add_argument("--dataset", required=True, help="Dataset name")
    p.add_argument("--limit", type=int, default=500, help="Number of queries")
    p.add_argument("--top_k", type=int, default=5, help="Top passages to use")
    p.add_argument("--secondary_dataset", default="truthfulqa")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--backend", choices=["openai", "ollama", "hf"], default="openai")
    args = p.parse_args()

    # Load dataset and retrieve contexts
    ds = load_dataset_file(args.dataset, args.limit, args.top_k)
    queries = ds["queries"]
    context_pool = ds.get("context_pool")

    retrieved = retrieve_top_k(RETRIEVER_MODEL, queries, top_k=args.top_k, context_pool=context_pool)

    prompts = [
        (item["query_id"], build_prompt(item["top_passages"], item["query"]))
        for item in retrieved
    ]

    if args.backend == "openai":
        preds = generate_openai(args.model_name, prompts, n_samples=1, temp=0.0)
    elif args.backend == "ollama":
        preds = generate_ollama(args.model_name, prompts)
    else:
        local_pipe = load_local_generator(args.model_name)
        preds = [
            {
                "query_id": qid,
                "answer": local_pipe(prompt)[0].get("generated_text", "").strip(),
            }
            for qid, prompt in tqdm(prompts, desc="Generating (HF)")
        ]

    ground_truth = build_ground_truth(queries)
    qa_metrics = evaluate_extractive_model(preds, ground_truth)

    # Hallucination
    judgement = judge_with_context(client, JUDGE_MODEL, retrieved, preds)

    idxs = random.sample(range(len(retrieved)), k=min(5, len(retrieved)))
    examples = [
        {
            "query_id": retrieved[i]["query_id"],
            "query": retrieved[i]["query"],
            "ground_truth": retrieved[i]["answers"],
            "prediction": preds[i]["answer"],
        }
        for i in idxs
    ]

    out = {
        "model_name": args.model_name,
        "evaluation_method": "RAG",
        "dataset": args.dataset,
        "num_queries": len(retrieved),
        "metrics": {"squad": qa_metrics, "hallucination": judgement},
        "examples": examples,
    }
    print(json.dumps(out, indent=2))

    save_evaluation_results(
        model_name=args.model_name,
        evaluation_method="RAG",
        dataset_name=args.dataset,
        squad_results={"exact_match": qa_metrics["exact_match"], "f1": qa_metrics["f1"]},
        rouge_results={k: qa_metrics[k] for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]},
        bleu_results={"bleu": qa_metrics["bleu"], "precisions": []},
        mean_metrics={},
        contextual_results=None,
        truth_metrics=judgement,
        examples=examples,
        num_entries=len(retrieved),
    )


if __name__ == "__main__":
    main()