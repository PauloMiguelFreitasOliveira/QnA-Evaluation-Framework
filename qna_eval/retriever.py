import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




def encode_queries(queries, tokenizer, model, device):
    embeddings = []
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                output = outputs.pooler_output.squeeze().cpu().numpy()
            else:
                output = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        embeddings.append(output / np.linalg.norm(output))
    return np.array(embeddings)


def encode_passages(passages, tokenizer, model, device):
    embeddings = []
    for passage in passages:
        inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                output = outputs.pooler_output.squeeze().cpu().numpy()
            else:
                output = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        embeddings.append(output / np.linalg.norm(output))
    return np.array(embeddings)

def is_relevant(passage, answers):
    return any(ans.lower() in passage.lower() for ans in answers)


def retrieve_top_k(model_name, queries, top_k, context_pool=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    model.eval()

    results = []
    for item in queries:
        query = item["query"]
        if context_pool:
            passages = context_pool
            candidates = [{"passage_text": p, "is_selected": int(is_relevant(p, item["answers"]))} for p in context_pool]
        else:
            candidates = item["candidates"]
            passages = [c["passage_text"] for c in candidates]

        query_emb = encode_queries([query], tokenizer, model, device)
        passage_embs = encode_passages(passages, tokenizer, model, device)

        scores = np.dot(passage_embs, query_emb.squeeze())
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_passages = [passages[i] for i in top_indices]

        # Build full passage score relevance list for contextual metric
        top_relevance = [
            {
                "text": passages[i],
                "score": float(scores[i]),
                "is_selected": int(candidates[i]["is_selected"])
            }
            for i in top_indices
        ]

        results.append({
            "query": query,
            "query_id": item["query_id"],
            "answers": item["answers"],
            "top_passages": top_passages,
            "scores": [float(scores[i]) for i in top_indices],
            "top_relevance": top_relevance,  # ✅ added
            "qrel": {str(i): int(candidates[i]["is_selected"]) for i in range(len(passages))},
            "run": {str(i): float(scores[i]) for i in range(len(passages))}
        })

    return results
