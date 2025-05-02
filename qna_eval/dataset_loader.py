# qna_eval/dataset_loader.py

import os
import json
import random
from datasets import load_dataset

def load_dataset_file(dataset_name, limit, max_contexts_per_query):
    dataset_path = os.path.join("datasets", f"{dataset_name}.json")


    if os.path.exists(dataset_path):
        print(f"📂 Loading dataset locally from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        print(f"🌐 Local file not found. Trying to load '{dataset_name}' from HuggingFace...")
        try:
            split_size = min(limit, 1000)
            split_string = f"train[:{split_size}]"
            raw_data = load_dataset(dataset_name, split=split_string)
        except Exception as e:
            raise ValueError(f" Dataset '{dataset_name}' not found locally or on HuggingFace. Error: {str(e)}")
        
    if isinstance(raw_data, dict):
        # Convert dictionary entries to a list of dicts with "id" field
        raw_data = [{"id": key, **value} for key, value in raw_data.items()]
        print(f"Raw data structure (limited): {raw_data[:5]}")

    elif isinstance(raw_data, dict):
        print(f"Raw data structure (limited): {dict(list(raw_data.items())[:5])}")

    formatted_data, context_pool = try_format_dataset(raw_data, limit=limit, max_contexts_per_query=max_contexts_per_query)
    
    # Ensure we're returning a valid dictionary with 'queries' key
    if formatted_data is None or context_pool is None:
        raise ValueError("❌ Error in formatting dataset: formatted_data or context_pool is None.")
    
    return {"queries": formatted_data, "context_pool": context_pool}





def try_format_dataset(raw_data, limit, max_contexts_per_query):
    formatted = []
    context_pool = []

    if isinstance(raw_data, list):
        entries = raw_data
    elif hasattr(raw_data, '__getitem__'): 
        entries = raw_data
    elif isinstance(raw_data, dict):  # SQuAD-style dictionary
        entries = [{"query_id": qid, **entry} for qid, entry in raw_data.items()]
    else:
        raise ValueError("❌ Unrecognized dataset structure.")

    count = 0
    for entry in entries:
        try:
            query_id = str(entry.get("query_id", entry.get("id", count)))
            query = entry.get("query", entry.get("question", None))
            context = entry.get("context", None)
            answers_raw = entry.get("answers", [])

            # Normalize answers
            answer_texts = []
            if isinstance(answers_raw, list):
                for a in answers_raw:
                    if isinstance(a, dict) and "text" in a:
                        answer_texts.append(a["text"])
                    elif isinstance(a, str):
                        answer_texts.append(a)
            elif isinstance(answers_raw, dict) and "text" in answers_raw:
                answer_texts.append(answers_raw["text"])

            if not query or not context:
                print(f"⚠️ Skipping entry {query_id}: Missing query or context.")
                continue

            # Save the context (passage) to the context pool for use later
            context_pool.append(context)

            # Save formatted entry with only its original context
            formatted.append({
                "query_id": query_id,
                "query": query,
                "answers": answer_texts,
                "candidates": [{
                    "passage_text": context,
                    "is_selected": bool(answer_texts),
                    "url": ""
                }]
            })

            count += 1
            if count >= limit:
                break

        except Exception as e:
            print(f"❌ Error formatting entry {count}: {str(e)}")
            continue

    if not formatted:
        raise ValueError("❌ Dataset could not be formatted: No valid entries found.")

    # Print the number of contexts in the context pool
    print(f"📦 Number of contexts in the context pool: {len(context_pool)}")

    # If the dataset has fewer than `max_contexts_per_query` contexts per query, 
    # use the context pool for ranking (this is for downstream logic)
    if len(context_pool) <= max_contexts_per_query:
        print("📎 Using context pool for ranking (less than 10 contexts per query).")

    num_answerable = sum(1 for q in formatted if q["answers"])
    print(f"✅ Formatted {len(formatted)} entries ({num_answerable} with answers, {len(formatted)-num_answerable} unanswerable).")

    return formatted, context_pool
