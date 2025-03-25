import json

def format_squad_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load the JSON file

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  # Pretty-print the JSON

# Usage
input_file = '../data/squad.json'  # Update with your file path
output_file = '../datasets/squad_formatted.json'
format_squad_json(input_file, output_file)

print("Formatted SQuAD JSON saved to:", output_file)
