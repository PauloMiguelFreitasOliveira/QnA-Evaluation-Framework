"""
Loads and formats datasets from local files or HuggingFace Hub.
Prepares queries, answers, and context passages for evaluation or retrieval tasks.
"""


import os
import json
from datasets import load_dataset
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Loads a dataset from local JSON or HuggingFace, then formats it for retrieval
def load_dataset_file(dataset_name, limit, max_contexts_per_query=None):
    dataset_path = os.path.join("datasets", f"{dataset_name}.json")

    if os.path.exists(dataset_path):
        print(f" Loading dataset locally from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        print(f" Local file not found. Trying to load '{dataset_name}' from HuggingFace...")
        try:
            split_size = min(limit, 1000)
            split_string = f"train[:{split_size}]"
            raw_data = load_dataset(dataset_name, split=split_string)
        except Exception as e:
            raise ValueError(f" Dataset '{dataset_name}' not found locally or on HuggingFace. Error: {str(e)}")
        
    if isinstance(raw_data, dict):
        # Convert dictionary entries to a list of dicts with "id" field
        raw_data = [{"id": key, **value} for key, value in raw_data.items()]

    formatted_data, context_pool = try_format_dataset(raw_data, limit=limit, max_contexts_per_query=max_contexts_per_query)
    
    # Ensure we're returning a valid dictionary with 'queries' key
    if formatted_data is None or context_pool is None:
        raise ValueError(" Error in formatting dataset: formatted_data or context_pool is None.")
    
    # ✅ Print up to 200 entries after deduplication
    #print("\n🖨️ Showing up to 200 unique formatted entries:\n")
    #for i, entry in enumerate(formatted_data[:200]):
    #    print(f"\n🔹 Entry {i+1}:")
    #    print(f"Query ID: {entry.get('query_id')}")
    #    print(f"Query: {entry.get('query')}")
    #    print(f"Answers: {entry.get('answers')}")
    #    print("Candidates:")
    #    for c in entry.get("candidates", []):
    #        print(f"  - Passage: {c.get('passage_text')[:200]}...")  # Truncate long text
    #        print(f"    Selected: {c.get('is_selected')}")
    
    return {"queries": formatted_data, "context_pool": context_pool}

# Converts raw dataset entries into standardized format with query, answers, and context
def try_format_dataset(raw_data, limit, max_contexts_per_query):
    formatted = []
    context_pool = []
    seen_entries = set()

    if isinstance(raw_data, list):
        entries = raw_data
    elif hasattr(raw_data, '__getitem__'): 
        entries = raw_data
    else:
        raise ValueError(" Unrecognized dataset structure.")

    count = 0
    for count, entry in enumerate(entries):
        ##print(f"\n🔍 Example {count}: {entry}")
        try:
            query_id = str(entry.get("query_id", entry.get("id", count)))
            query = entry.get("query", entry.get("question", None))
            answers_raw = entry.get("answers", [])

                # Parse answer/context if embedded in "text" field
            if not answers_raw or entry.get("context") is None:
                text_field = entry.get("text", "")
                answer_match = re.search(r"<answer>\s*(.*?)\s*<context>", text_field, re.DOTALL)
                context_match = re.search(r"<context>\s*(.*)", text_field, re.DOTALL)

                if answer_match:
                    answers_raw = [answer_match.group(1).strip()]
                if context_match:
                    entry["context"] = context_match.group(1).strip()

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
                selected = [c["passage_text"] for c in candidates if c.get("is_selected", 0)]
                if selected:
                    context = selected[0]

            if not query or not context:
                #print(f" Skipping entry {query_id}: Missing query or context.")
                continue

                        # Deduplication: check if (query, context, answer) is unique
            is_duplicate = False
            for ans in answer_texts or [""]:
                key = (query.strip(), context.strip(), ans.strip())
                if key in seen_entries:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Add to seen set
            for ans in answer_texts or [""]:
                key = (query.strip(), context.strip(), ans.strip())
                seen_entries.add(key)


            # Only create the context pool if there are 1–3 passages per query
            if (candidates and len(candidates) <= 3) or (context and not candidates):
                if candidates:
                    selected_contexts = [c["passage_text"] for c in candidates[:max_contexts_per_query]]
                    context_pool.extend([c for c in selected_contexts if c not in context_pool])
                else:
                    if context not in context_pool:
                        context_pool.append(context)

            # Save formatted entry with its selected context (or fallback context)
            formatted.append({
                "query_id": query_id,
                "query": query,
                "answers": answer_texts,
                "candidates": [
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
            print(f" Error formatting entry {count}: {str(e)}")
            continue

    if not formatted:
        raise ValueError(" Dataset could not be formatted: No valid entries found.")

    print(f" Number of contexts in the context pool: {len(context_pool) if context_pool else 0}")
    num_answerable = sum(1 for q in formatted if q["answers"])
    print(f" Formatted {len(formatted)} entries ({num_answerable} with answers, {len(formatted)-num_answerable} unanswerable).")

    return formatted, context_pool
