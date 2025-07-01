# frontend_utils.py

import json
import os

import streamlit as st

PROCESS_EXPLANATIONS = {
    "Loading dataset": (
        "Loading the evaluation dataset into memory. Reads all questions and their corresponding context passages, "
        "validates data structure, and ensures the dataset is ready for evaluation."
    ),
    "Loading primary dataset": (
        "Loading the main evaluation dataset, which contains the primary set of questions and context passages that will be used "
        "to assess model performance."
    ),
    "Loading secondary dataset": (
        "Loading an additional dataset—typically for hallucination or factual consistency checks. This dataset (often TruthfulQA) "
        "is used to challenge models with questions that can reveal overconfidence or fabricated answers."
    ),
    "Retrieving top-K contexts": (
        "For each question, the retriever model scans the available context pool and selects the top-K passages most relevant to the query. "
        "This is a crucial step because the model’s answer quality depends heavily on the relevance of these retrieved passages."
    ),
    "Contexts retrieved": (
        "The retrieval step is complete. The most relevant passages for each question have been identified and will be passed to the reader or generative model for answer generation."
    ),
    "Evaluating retrieval metrics": (
        "Computing information retrieval metrics such as Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (nDCG), and Mean Reciprocal Rank (MRR). "
        "These metrics evaluate how well the retriever surfaced relevant passages for the questions."
    ),
    "Finished retrieval metrics": (
        "All retrieval metrics have been calculated, giving a quantitative sense of how effective the retriever is at finding relevant information."
    ),
    "Loading reader model": (
        "Loading the extractive question-answering (reader) model. This could be a neural network or transformer-based model that will attempt to find answers within the retrieved contexts."
    ),
    "Extracting answers with reader": (
        "The reader model is now analyzing each question and its associated retrieved passages, and generating the best answer span(s) it can extract from the given contexts."
    ),
    "Evaluating QA metrics": (
        "Evaluating the model’s answers using SQuAD-style metrics such as Exact Match (EM) and F1, as well as ROUGE and BLEU. "
        "These metrics measure how closely the model’s answers match the human-annotated ground truth."
    ),
    "Finished all QA evaluations": (
        "All question-answering metrics have been computed. The results summarize the model’s accuracy and overlap with the ground-truth answers."
    ),
    "Saving results": (
        "Saving all evaluation outputs—including summary statistics, detailed logs, and sample answers—to disk for later analysis and dashboard visualization."
    ),
    "Generating primary answers": (
        "The selected generative model is creating answers for each question in the main (primary) evaluation dataset. "
        "Depending on the model, this might involve advanced reasoning, synthesis, or summarization capabilities."
    ),
    "Generating secondary predictions (hallucination eval)": (
        "The model is being challenged with questions from a secondary dataset (such as TruthfulQA) to probe its ability to provide factually consistent answers. "
        "This helps in assessing how often the model may hallucinate or invent information."
    ),
    "Building prompts": (
        "For each question, constructing a natural-language prompt that provides both the retrieved context and the question itself. "
        "These prompts are designed to guide generative models (like GPT, T5, or Llama) to generate answers using only the given information."
    ),
    "Generating answers": (
        "The generative model (such as an LLM) is now producing answers based on the constructed prompts. "
        "This step may involve advanced reasoning and text generation—models generate free-form answers, leveraging both the question and provided context."
    ),
    "Judging hallucinations": (
        "A specialized fact-checking (judge) model is reviewing each generated answer, verifying whether it is fully supported by the provided context. "
        "This step is crucial for identifying answers that may be partially or entirely hallucinated."
    ),
    "Preparing examples and output": (
        "Compiling sample questions, model predictions, and ground-truth answers into a summary for user inspection. "
        "This helps users qualitatively assess model behavior in addition to the numeric metrics."
    ),
    "Preparing output": (
        "Organizing and formatting all evaluation outputs—metrics, sample answers, and logs—for display and permanent storage."
    ),
    "Evaluation Complete": (
        "The evaluation run has finished! All results, logs, and summary statistics are now available for review."
    ),
}



METRIC_EXPLANATIONS = {
    "Exact Match": "Percentage of answers that exactly match the ground truth.",
    "F1": "The harmonic mean of precision and recall for the predicted answers.",
    "ROUGE-L": "Longest Common Subsequence metric, measuring overlap between prediction and reference.",
    "ROUGE1": "Overlap of unigrams between prediction and ground truth.",
    "ROUGE2": "Overlap of bigrams between prediction and ground truth.",
    "ROUGEL": "Longest Common Subsequence, another variant.",
    "ROUGELSUM": "Summarization variant of ROUGE-L.",
    "BLEU": "Bilingual Evaluation Understudy; measures n-gram overlap between prediction and ground truth.",
    "MAP": "Mean Average Precision: measures the precision of the retriever at all ranks.",
    "nDCG": "Normalized Discounted Cumulative Gain: rewards higher-ranked relevant results.",
    "Prec. @ K": "Average precision of the top-K retrieved passages.",
    "PHR": "Partial Hallucination Rate: % of answers partially supported by the context.",
    "THR": "Total Hallucination Rate: % of answers completely unsupported by the context.",
}


def display_metrics_with_explanations(metrics_dict):
    for metric, value in metrics_dict.items():
        label = metric.replace("_", " ").title()
        if label.upper() in METRIC_EXPLANATIONS:
            expl = METRIC_EXPLANATIONS[label.upper()]
        else:
            expl = METRIC_EXPLANATIONS.get(label, "")
        with st.expander(f"{label}: {value}"):
            st.write(expl)

def safe_avg_precision(precisions):
    return sum(precisions) / len(precisions) if precisions else 0

def is_noisy(line):
    noise_patterns = [
        "This IS expected",
        "This IS NOT expected",
        "Some weights of the model checkpoint",
        "Device set to use cpu",
        "Downloading",
        "Loading weights",
    ]
    return any(pat in line for pat in noise_patterns)

def append_to_results(new_entry, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(new_entry)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def is_progress_json(line):
    try:
        j = json.loads(line)
        return "process_stage" in j and "progress" in j and len(j) == 2
    except Exception:
        return False

def is_final_result_json(line):
    try:
        j = json.loads(line)
        return "metrics" in j
    except Exception:
        return False
