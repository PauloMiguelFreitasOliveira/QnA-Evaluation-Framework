import json

def convert_top1000_to_json(input_file, output_file):
    top1000_data = []

    # Open the input file with explicit encoding
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line
    for line in lines:
        # Strip any leading/trailing whitespace and split by tab
        parts = line.strip().split('\t')
        
        if len(parts) >= 4:  # Ensure there are enough parts
            doc_id = parts[0]
            query_id = parts[1]
            query = parts[2]
            context = ' '.join(parts[3:])  # Combine the remaining parts as the context text
            
            # Create a dictionary for the current entry
            entry = {
                'doc_id': doc_id,
                'query_id': query_id,
                'query': query,
                'context': context  # Store the context instead of answer
            }

            # Append the entry to the data list
            top1000_data.append(entry)
        else:
            # If the line is incorrectly formatted, print and skip it
            print(f"Skipping line (not enough parts): {line.strip()}")

    # Write the data to the output file in JSON format
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(top1000_data, out_file, ensure_ascii=False, indent=4)

# Usage
input_file = '../data/top1000.dev'
output_file = '../datasets/top1000.json'
convert_top1000_to_json(input_file, output_file)
