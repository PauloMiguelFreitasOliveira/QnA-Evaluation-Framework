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


def retrieve_top_k(model_name, queries, top_k=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    model.eval()

    results = []
    for item in queries:
        query = item["query"]
        passages = [c["passage_text"] for c in item["candidates"]]

        query_emb = encode_queries([query], tokenizer, model, device)
        passage_embs = encode_passages(passages, tokenizer, model, device)

        scores = np.dot(passage_embs, query_emb.squeeze())
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_passages = [passages[i] for i in top_indices]

        results.append({
            "query": query,
            "query_id": item["query_id"],
            "answers": item["answers"],
            "top_passages": top_passages,
            "scores": [float(scores[i]) for i in top_indices],
            "qrel": {str(i): int(item["candidates"][i]["is_selected"]) for i in range(len(passages))},
            "run": {str(i): float(scores[i]) for i in range(len(passages))}
        })

    return results
