"""
Runs extractive QA evaluation using SQuAD metrics, BLEU, and ROUGE.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

# Load SQuAD metric for extractive QA
squad_metric = load("squad")

# Performs extractive QA inference using a huggingface reader.
def extract_answer(model, tokenizer, question, context, device="cpu"):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    if start <= end:
        tok_ids = inputs.input_ids[0][start : end+1]
        return tokenizer.decode(tok_ids, skip_special_tokens=True)
    return ""

# Main extractive QA evaluation: SQuAD (EM/F1), BLEU, and ROUGE.
def evaluate_extractive_model(predictions, ground_truth_dict):
    preds_for_metric = [
        {"id": p["query_id"], "prediction_text": p["answer"]}
        for p in predictions
    ]
    refs_for_metric = []
    for p in predictions:
        gold_answers = ground_truth_dict.get(p["query_id"], [])
        refs_for_metric.append({
            "id": p["query_id"],
            "answers": {
                "text": gold_answers,
                "answer_start": [0] * len(gold_answers)
            }
        })

    squad_results = squad_metric.compute(
        predictions=preds_for_metric,
        references=refs_for_metric
    )
    em = squad_results.get("exact_match", 0.0)
    f1 = squad_results.get("f1", 0.0)

    # BLEU
    smooth = SmoothingFunction().method4
    bleu_scores = []
    for p in predictions:
        gt_texts = ground_truth_dict.get(p["query_id"], [])
        references = [word_tokenize(ref.lower()) for ref in gt_texts if ref.strip()]
        hypothesis = word_tokenize(p["answer"].lower())

        if references and hypothesis:
            bleu_scores.append(
                sentence_bleu(references, hypothesis, smoothing_function=smooth)
            )
        else:
            bleu_scores.append(0.0)
        
    bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    # ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    rouge1, rouge2, rougeL, rougeLsum = 0.0, 0.0, 0.0, 0.0
    for p in predictions:
        references = ground_truth_dict.get(p["query_id"], [])
        ref_text = references[0] if references else ""
        scores = rouge_scorer_obj.score(p["answer"], ref_text)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure
        rougeLsum += scores["rougeLsum"].fmeasure
    n = len(predictions)
    rouge1 = rouge1 / n if n else 0.0
    rouge2 = rouge2 / n if n else 0.0
    rougeL = rougeL / n if n else 0.0
    rougeLsum = rougeLsum / n if n else 0.0

    # Assemble final metrics
    metrics = {
        "exact_match": round(em, 2),
        "f1":          round(f1, 2),
        "bleu":        round(bleu, 4),
        "rouge1":      round(rouge1, 4),
        "rouge2":      round(rouge2, 4),
        "rougeL":      round(rougeL, 4),
        "rougeLsum":   round(rougeLsum, 4)
    }
    return metrics
