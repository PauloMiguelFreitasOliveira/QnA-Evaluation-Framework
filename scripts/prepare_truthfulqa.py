import json
import os
from datasets import load_dataset

# Convert the Domenic Rosati TruthfulQA dataset (free-text QA) into your framework format

def prepare_truthfulqa(
    output_path: str = "datasets/truthfulqa.json",
    limit: int = 817
):
    """
    Loads the Domenic Rosati TruthfulQA dataset, extracts question and correct answers,
    and writes to JSON in your format:
      - query_id
      - query
      - answers (list of strings)
      - candidates: a single dummy entry with empty passage_text

    Stops after `limit` valid entries.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load the dataset
    try:
        ds = load_dataset("domenicrosati/TruthfulQA", split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load Domenicrosati TruthfulQA: {e}")

    formatted = []
    for idx, ex in enumerate(ds):
        # Extract question
        question = ex.get("Question") or ex.get("question")
        if not question or not question.strip():
            continue
        question = question.strip()

        # Extract answers from 'Correct Answers' or 'Best Answer'
        answers = []
        raw_correct = ex.get("Correct Answers") or ex.get("correct_answers")
        if isinstance(raw_correct, str) and raw_correct.strip():
            answers = [ans.strip() for ans in raw_correct.split(";") if ans.strip()]

        # Fallback to Best Answer
        if not answers:
            best = ex.get("Best Answer") or ex.get("best_answer")
            if isinstance(best, str) and best.strip():
                answers = [best.strip()]

        if not answers:
            continue

        formatted.append({
            "query_id": str(idx),
            "query": question,
            "answers": answers,
            "candidates": [
                {"passage_text": "", "is_selected": True, "url": ""}
            ]
        })

        if len(formatted) >= limit:
            break

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(formatted)} TruthfulQA entries to {output_path}")


if __name__ == "__main__":
    prepare_truthfulqa()
