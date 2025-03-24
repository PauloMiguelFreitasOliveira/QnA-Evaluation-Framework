import json

# Read the queries file
def convert_queries_to_json(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # If utf-8 fails, try ISO-8859-1 (latin1)
        with open(input_file, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
    queries_data = []

    # Iterate through each line in queries file
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            query_id, query_text = parts
            queries_data.append({
                "query_id": query_id,
                "query_text": query_text
            })

    # Write to a json file
    with open(output_file, 'w') as f:
        json.dump(queries_data, f, indent=4)

# Usage
input_file = '../data/queries.eval.tsv'
output_file = '../datasets/queries.json'
convert_queries_to_json(input_file, output_file)
