import json
from datasets import load_dataset

# Load MS MARCO v2.1
dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")

output = []

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

# Save to ms_marco.json
with open("..\datasets\ms_marco.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f" Saved {len(output)} entries to ms_marco.json")
