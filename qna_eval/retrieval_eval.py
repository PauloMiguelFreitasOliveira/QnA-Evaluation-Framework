import json
import pytrec_eval
import numpy as np


# Load qrels
with open('../datasets/qrels.json', 'r', encoding='utf-8') as f:
    qrels_raw = json.load(f)

# Convert qrels to pytrec_eval format
qrels = {}
for query_id, docs in qrels_raw.items():
    qrels[query_id] = {}
    for doc in docs:
        qrels[query_id][doc['doc_id']] = int(doc['relevance'])

# Load top1000 results (BM25-ranked)
with open('../datasets/top1000.json', 'r', encoding='utf-8') as f:
    top1000_raw = json.load(f)

# Convert top1000 to pytrec_eval format (run)
# We assign scores in reverse order (top doc gets highest score)
run = {}
for query_id, docs in top1000_raw.items():
    run[query_id] = {}
    for rank, doc in enumerate(docs):
        # Score is inverse rank: higher score = better ranking
        score = len(docs) - rank
        run[query_id][doc['doc_id']] = score

matched_queries = 0
for qid in run:
    if qid in qrels:
        retrieved = set(run[qid].keys())
        relevant = set(qrels[qid].keys())
        if retrieved & relevant:
            matched_queries += 1

print(f"✅ Queries with at least one relevant doc retrieved: {matched_queries}")



overlap = set(qrels.keys()) & set(run.keys())
print(f"\n🧩 Queries in both QREL and RUN: {len(overlap)}")


# Filter queries present in both qrels and run
valid_query_ids = set(qrels.keys()).intersection(run.keys())
filtered_qrels = {qid: qrels[qid] for qid in valid_query_ids}
filtered_run = {qid: run[qid] for qid in valid_query_ids}

print("🔍 Sample QREL:")
for qid in list(qrels.keys())[:3]:
    print(f"{qid}: {qrels[qid]}")

print("\n📥 Sample RUN:")
for qid in list(run.keys())[:3]:
    print(f"{qid}: {list(run[qid].items())[:3]}")

# Evaluate using pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, {'map', 'ndcg', 'recip_rank'})
results = evaluator.evaluate(filtered_run)

# Average metrics across queries
mean_metrics = {
    metric: np.mean([query_measures[metric] for query_measures in results.values()])
    for metric in ['map', 'ndcg', 'recip_rank']
}

# Output results
print("📊 BM25 Ranking Evaluation (from top1000)")
print(f"Mean Average Precision (MAP):     {mean_metrics['map']:.4f}")
print(f"Normalized DCG (NDCG):            {mean_metrics['ndcg']:.4f}")
print(f"Mean Reciprocal Rank (MRR):       {mean_metrics['recip_rank']:.4f}")
