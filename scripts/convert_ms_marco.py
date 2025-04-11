import json
from datasets import load_dataset

# Load only the first 100,000 entries from MS MARCO v2.1 train split
dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train[:100000]")

output = []

# Iterate through the subset and collect data
for entry in dataset:
    query_id = str(entry["query_id"])
    query = entry["query"]
    answers = entry["answers"]

    candidates = []
    for text, selected, url in zip(entry["passages"]["passage_text"],
                                   entry["passages"]["is_selected"],
                                   entry["passages"]["url"]):
        candidates.append({
            "passage_text": text,
            "is_selected": selected,
            "url": url
        })

    output.append({
        "query_id": query_id,
        "query": query,
        "answers": answers,
        "candidates": candidates
    })

# Save to ../datasets/ms_marco.json
output_path = "../datasets/ms_marco.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Successfully saved {len(output)} entries to {output_path}")
