import json
import os

# Define paths
results_path = 'evaluation_results.json'
output_path = 'evaluation_results.json'

# Load your file (relative to current working dir)
with open(results_path, encoding='utf-8') as f:
    data = json.load(f)

filtered = []
for entry in data:
    to_check = []
    # Collect all relevant counts
    if 'num_entries' in entry:
        to_check.append(entry['num_entries'])
    if 'num_primary' in entry:
        to_check.append(entry['num_primary'])
    if 'num_secondary' in entry:
        to_check.append(entry['num_secondary'])
    # If any count exists and is < 100, exclude
    if to_check and any((val is not None and val < 100) for val in to_check):
        continue
    # Otherwise keep
    filtered.append(entry)

# Save output to a *new* file, not to overwrite original
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Filtered list saved as {output_path}. Total kept: {len(filtered)}")