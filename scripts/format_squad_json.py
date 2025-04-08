import json

# Format the SQuAD JSON for later use
def format_squad_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load the original SQuAD JSON

    squad_dict = {}

    # Iterate through the data and structure it into a dictionary
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                squad_dict[qa["id"]] = {
                    "question": qa["question"],
                    "answers": qa["answers"],
                    "context": paragraph["context"]
                }

    # Write the structured data into a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_dict, f, indent=4, ensure_ascii=False)

# Usage
input_file = '../datasets/squad.json'
output_file = '../datasets/squad_formatted.json'
format_squad_json(input_file, output_file)
