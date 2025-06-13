import os
import json
from openai import OpenAI

def load_secondary_dataset(name: str, limit: int):
    """
    Load the generative‐only QA JSON (datasets/<name>.json), keeping only entries
    with a nonempty 'query'. Returns up to `limit` entries.
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


def generate_multisample(
    model_pipe,
    queries: list[dict],
    prompt_tpl: str,
    n_samples: int = 10,
    **gen_kwargs
) -> list[dict]:
    """
    For each query, call `model_pipe` n_samples times and return a flat list
    of {"query_id","answer"} dicts.
    """
    outs = []
    for ex in queries:
        prompt = prompt_tpl.format(question=ex["query"])
        for _ in range(n_samples):
            out = model_pipe(prompt, **gen_kwargs)[0]
            text = out.get("generated_text") or out.get("text") or ""
            outs.append({
                "query_id": ex["query_id"],
                "answer": text.strip()
            })
    return outs


def judge_with_llm(
    client: OpenAI,
    model_name: str,
    queries: list[dict],
    samples: list[dict],
    fs_examples: list[dict] | None = None
) -> dict[str, float]:
    """
    Use an LLM to judge each generated answer as SUPPORT or HALLUCINATION.
    Computes:
      - PHR = fraction of queries where at least one sample is hallucinated
      - THR = fraction of queries where all samples are hallucinated
    `queries` is list of {"query_id","query","answers"}.
    `samples` is flat list of {"query_id","answer"}.
    `fs_examples` is an optional few-shot list of messages for the judge.
    """
    # group all samples by query_id
    by_q = {}
    for s in samples:
        by_q.setdefault(s["query_id"], []).append(s["answer"])

    total_q = len(queries)
    phr = 0
    thr = 0

    for ex in queries:
        qid = ex["query_id"]
        refs = ex["answers"]
        ans_list = by_q.get(qid, [])

        flags = []
        for ans in ans_list:
            # build messages
            msgs = []
            if fs_examples:
                msgs.extend(fs_examples)
            msgs.append({
                "role": "system",
                "content": (
                    "You are a fact-checker. "
                    "Given a question, reference answers, and a candidate answer, "
                    "reply exactly SUPPORT if the candidate is fully supported, "
                    "or HALLUCINATION otherwise."
                )
            })
            ref_block = "\n".join(f"- {r}" for r in refs[:5])
            msgs.append({
                "role": "user",
                "content": (
                    f"Question: {ex['query']}\n\n"
                    f"Reference answer(s):\n{ref_block}\n\n"
                    f"Candidate answer:\n{ans}\n\n"
                    "Is this candidate fully supported by the reference answer(s)?"
                )
            })

            resp = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.0,
                max_tokens=3
            )
            verdict = resp.choices[0].message.content.strip().upper()
            flags.append(verdict.startswith("SUPPORT"))

        # if any sample is hallucinated → count toward PHR
        if any(not f for f in flags):
            phr += 1
        # if all samples hallucinated → count toward THR
        if all(not f for f in flags):
            thr += 1

    return {
        "PHR": round(phr / total_q, 4),
        "THR": round(thr / total_q, 4)
    }
