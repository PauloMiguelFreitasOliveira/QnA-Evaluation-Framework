# LLM Question Answering Evaluation Framework

A modular framework for evaluating and benchmarking Question Answering (QA) systems built on Large Language Models (LLMs).

This project was developed as part of my Master's Dissertation in Computer Engineering and provides a unified environment for comparing Retrieval-Augmented Generation (RAG), Extractive QA, and Generative QA approaches using multiple evaluation metrics.

---

## Features

- Modular evaluation pipeline
- Retrieval benchmarking (BM25, DPR and Hybrid Retrieval)
- Extractive Question Answering evaluation
- Generative Question Answering evaluation
- Retrieval-Augmented Generation (RAG) evaluation
- Hallucination-aware evaluation
- Interactive Streamlit interface
- Automated experiment execution and result comparison

---

## Technologies

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- OpenAI
- FAISS
- DPR
- BM25
- Pandas
- JSON

---

## Project Structure

```
Dataset
      │
      ▼
Retriever
      │
      ▼
Reader / LLM
      │
      ▼
Evaluation Pipeline
      │
      ▼
Metrics & Analysis
```

---

## Motivation

Evaluating Large Language Models consistently is a challenging task due to the variety of architectures, retrieval methods and evaluation metrics available.

This framework provides a reusable and extensible environment for benchmarking different QA systems using standardized datasets and evaluation metrics, making it easier to compare model performance and support research or enterprise AI development.

---

## Learning Outcomes

Through this project I gained practical experience with:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- AI Evaluation Metrics
- Prompt Engineering
- Python Software Engineering
- Streamlit Applications
- Experimental AI Research

---

## Author

Paulo Oliveira

## Installation

Use `pip` to install all dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

1. Place your dataset under `datasets/<name>.json` or rely on a dataset from HuggingFace.
2. Run one of the evaluation scripts shown below.
3. Metrics and sample outputs are appended to `qna_eval/results/evaluation_results.json`.

### Metrics

- **Retrieval**: mean average precision (MAP), NDCG and MRR.
- **Extractive QA**: SQuAD exact match and F1, ROUGE-1/2/L/Lsum and BLEU.
- **Hallucination**: PHR and THR measured with an LLM judge when running generative evaluation.

## Scripts

### `retrieval_eval.py`

Evaluates a retriever and optional reader model on a dataset. Example:

```bash
python retrieval_eval.py \
  --retriever_model bm25 \
  --reader_model timpal0l/mdeberta-v3-base-squad2 \
  --dataset ms_marco \
  --top_k 10 \
  --limit 100
```

This loads the dataset, retrieves top passages, optionally runs the reader to generate answers and computes all metrics. Results are written to `qna_eval/results/evaluation_results.json`.

### `generative_eval.py`

Generates answers using a model and evaluates hallucination rates. Example:

```bash
python generative_eval.py \
  --model_name gpt-3.5-turbo \
  --primary_dataset ms_marco \
  --limit 500 \
  --n_samples 5 \
  --backend openai
```

The script first produces answers for the primary dataset then judges hallucinations on a secondary dataset (default `truthfulqa`).

## Results

After each run, a summary including metrics and random examples is saved in `qna_eval/results/evaluation_results.json`.
