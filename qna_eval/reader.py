from transformers import pipeline
import torch


def load_reader(model_name):
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)
    return qa_pipeline

def extract_answers(reader_pipeline, retrieval_results, separator=" [SEP] "):
    predictions = []
    for item in retrieval_results:
        question = item["query"]
        top_passages = item["top_passages"][:5]  # Top 5 passages
        combined_context = separator.join(top_passages)

        result = reader_pipeline(question=question, context=combined_context)
        answer_text = result.get("answer", "")

        predictions.append({
            "query_id": item["query_id"],
            "answer": answer_text
        })
    return predictions
