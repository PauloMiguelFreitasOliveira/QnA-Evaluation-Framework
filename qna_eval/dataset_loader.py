import os
import json
from datasets import load_dataset


DATASET_DIR = os.path.join(os.path.dirname(__file__),"../datasets")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(data, file_path):
    """Saving JSON file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def preprocess_squad(data):
    """Preprocess SQuAD dataset"""
    processed_data = []
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                processed_data.append({
                    "question": qa["question"],
                    "answer": qa["answer"][0]["text"] if qa["answers"] else "",
                    "context": context
                })
    return processed_data

def preprocess_ms_marco(data):
    """Preprocess MS MARCO dataset"""
    processed_data = [
        {"query": item["query"], "passage": item["passage"], "answer": item.get("answers", [""])[0]}
        for item in data
    ]
    return processed_data

def load_and_preprocess():
    """Loads, preprocesses, and saves datasets for future use"""
    datasets = {
        "squad": "squad.json",
        "ms_marco": "ms_marco.json"
    }

    for name, file in datasets.items():
        raw_path = os.path.join(DATASET_DIR, file)
        processed_path = os.path.join(PROCESSED_DIR, f"{name}_preprocessed.json")

        if os.path.exists(processed_path):
            print(f"Loading preprocessed {name} dataset...")
            dataset = load_json(processed_path)
        else:
            print(f"Preprocessing {name} dataset...")
            raw_data = load_json(raw_path)
            if name == "squad":
                dataset = preprocess_squad(raw_data)
            elif name == "ms_marco":
                dataset = preprocess_ms_marco(raw_data)
            save_json(dataset, processed_path)

        print(f"{name} dataset loaded with {len(dataset)} entries.")

    return dataset

if __name__ == "__main__":
    load_and_preprocess()        