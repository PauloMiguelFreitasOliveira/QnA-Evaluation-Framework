# qna_eval/dataset_loader.py

import os
import json
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
            split_size = min(limit, 1000)  # Safety: Don't over-fetch
            split_string = f"train[:{split_size}]"
            raw_data = load_dataset(dataset_name, split=split_string)
        except Exception as e:
            raise ValueError(f"❌ Dataset '{dataset_name}' not found locally or on HuggingFace. Error: {str(e)}")

    formatted_data = try_format_dataset(raw_data, limit=limit, max_contexts_per_query=max_contexts_per_query)
    return {"queries": formatted_data}

def try_format_dataset(raw_data, limit, max_contexts_per_query):
    formatted = []

    if isinstance(raw_data, list):
        entries = raw_data
    elif hasattr(raw_data, '__getitem__'):  # HuggingFace Dataset object
        entries = raw_data
    else:
        raise ValueError("❌ Unrecognized dataset structure.")

    count = 0
    for entry in entries:
        try:
            query_id = str(entry.get("query_id", entry.get("id", count)))
            query = entry.get("query", entry.get("question", None))
            answers = entry.get("answers", entry.get("answer", []))

            if isinstance(answers, dict) and "text" in answers:
                answers = answers["text"]

            candidates_raw = entry.get("candidates", None)
            if not candidates_raw and "passages" in entry:
                candidates_raw = [
                    {
                        "passage_text": p,
                        "is_selected": s,
                        "url": u
                    }
                    for p, s, u in zip(
                        entry["passages"]["passage_text"],
                        entry["passages"]["is_selected"],
                        entry["passages"].get("url", [""] * len(entry["passages"]["passage_text"]))
                    )
                ]

            if not query or not candidates_raw or answers is None:
                print(f"⚠️ Skipping entry {count}: Missing query, candidates or answers.")
                continue

            candidates = candidates_raw[:max_contexts_per_query]

            formatted.append({
                "query_id": query_id,
                "query": query,
                "answers": answers if isinstance(answers, list) else [answers],
                "candidates": candidates
            })

            count += 1
            if count >= limit:
                break

        except Exception as e:
            print(f"⚠️ Error formatting entry {count}: {str(e)}")
            continue

    if not formatted:
        raise ValueError("❌ Dataset could not be formatted: No valid entries found.")

    print(f"✅ Formatted {len(formatted)} queries successfully.")
    return formatted
