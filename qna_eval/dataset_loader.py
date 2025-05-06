"""
Loads and formats datasets from local files or HuggingFace Hub.
Prepares queries, answers, and context passages for evaluation or retrieval tasks.
"""


import os
import json
import random
from datasets import load_dataset

# Loads a dataset from local JSON or HuggingFace, then formats it for retrieval
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

# Converts raw dataset entries into standardized format with query, answers, and context
def try_format_dataset(raw_data, limit, max_contexts_per_query):
    formatted = []
    context_pool = []

    if isinstance(raw_data, list):
        entries = raw_data
    elif hasattr(raw_data, '__getitem__'): 
        entries = raw_data
    elif isinstance(raw_data, dict): 
        entries = [{"query_id": qid, **entry} for qid, entry in raw_data.items()]
    else:
        raise ValueError("❌ Unrecognized dataset structure.")

    count = 0
    for entry in entries:
        try:
            query_id = str(entry.get("query_id", entry.get("id", count)))
            query = entry.get("query", entry.get("question", None))
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

            # Get context: either from `context` (SQuAD) or from selected candidate (MS MARCO)
            context = entry.get("context", None)
            candidates = entry.get("candidates", [])

            if context is None and candidates:
                # Find selected passage(s)
                selected = [c["passage_text"] for c in candidates if c.get("is_selected", 0)]
                if selected:
                    context = selected[0]  # Pick first selected passage

            if not query or not context:
                print(f"⚠️ Skipping entry {query_id}: Missing query or context.")
                continue

            # Only create the context pool if there are 1–3 passages per query
            if len(candidates) <= 3:
                # Populate context pool with the first few passages (1–3 max)
                selected_contexts = [c["passage_text"] for c in candidates[:max_contexts_per_query]]
                context_pool.extend(selected_contexts)  # Add selected passages

            # Save formatted entry with its selected context (or fallback context)
            formatted.append({
                "query_id": query_id,
                "query": query,
                "answers": answer_texts,
                "candidates": [  # Include all candidates for downstream evaluation
                    {
                        "passage_text": c.get("passage_text", ""),
                        "is_selected": bool(c.get("is_selected", False)),
                        "url": c.get("url", "")
                    }
                    for c in candidates
                ] if candidates else [{
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

    # If we have more than 3 passages per entry, return None as the context pool
    if len(context_pool) > 0 and len(context_pool) > 3:
        context_pool = None  # No context pool to evaluate ranking correctly

    # If context_pool was not populated, print a message about it
    if not context_pool:
        print("⚠️ No valid context pool to evaluate (either empty or more than 3 passages per query).")

    if not formatted:
        raise ValueError("❌ Dataset could not be formatted: No valid entries found.")

    print(f"📦 Number of contexts in the context pool: {len(context_pool) if context_pool else 0}")
    num_answerable = sum(1 for q in formatted if q["answers"])
    print(f"✅ Formatted {len(formatted)} entries ({num_answerable} with answers, {len(formatted)-num_answerable} unanswerable).")

    return formatted, context_pool
