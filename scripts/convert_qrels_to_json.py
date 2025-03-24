import json

# Read the qrels file
def convert_qrels_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    qrels_data = []

    # Iterate through each line in qrels file
    for line in lines:
        # Split by tabs
        parts = line.strip().split('\t')
        if len(parts) == 4:
            query_id, _, doc_id, relevance = parts
            qrels_data.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": int(relevance)
            })

    # Write to a json file
    with open(output_file, 'w') as f:
        json.dump(qrels_data, f, indent=4)

# Usage
input_file = '../data/qrels.dev.tsv'
output_file = '../datasets/qrels.json'
convert_qrels_to_json(input_file, output_file)
