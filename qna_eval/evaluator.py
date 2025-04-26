import pytrec_eval
import numpy as np
from .extractive_eval import evaluate_extractive_model

def evaluate_qa(queries, predictions, model_name):
    gt = {q["query_id"]: q["answers"] for q in queries}
    return evaluate_extractive_model(predictions, gt, model_name)

def evaluate_retrieval(qrel, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'recip_rank'})
    retrieval_metrics = evaluator.evaluate(run)

    mean_metrics = {
        metric: np.mean([query_measures[metric] for query_measures in retrieval_metrics.values()])
        for metric in ['map', 'ndcg', 'recip_rank']
    }

    return mean_metrics
