import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load the preprocessed JSON datasets
queries_dict = load_json('datasets/queries.json')
qrels_dict = load_json('datasets/qrels.json')
top1000_dict = load_json('datasets/top1000.json')

# Function to test lookups for queries, relevant documents, and retrieved passages
def test_lookup(query_id):
    # Fetch the query text from the queries_dict
    query_text = queries_dict.get(query_id)
    if query_text is None:
        print(f"❌ Query ID {query_id} not found!")
        return

    print(f"🔎 Query: {query_text}")

    # Fetch the relevant documents for the given query ID from qrels_dict
    relevant_docs = qrels_dict.get(query_id, [])
    if not relevant_docs:
        print(f"❌ No relevant documents found for Query ID {query_id}")
    else:
        print(f"✅ Relevant Docs:")
        for doc in relevant_docs:
            doc_id = doc['doc_id']
            relevance = doc['relevance']
            print(f" - Doc ID: {doc_id} | Relevance: {relevance}")

    # Fetch the retrieved passages for the given query ID from top1000_dict
    retrieved_passages = top1000_dict.get(query_id, [])
    if not retrieved_passages:
        print(f"❌ No retrieved passages found for Query ID {query_id}")
    else:
        print(f"✅ Retrieved Passages:")
        for passage in retrieved_passages:
            doc_id = passage['doc_id']
            passage_text = passage['passage']
            print(f" - Doc ID: {doc_id} | Passage: {passage_text[:80]}...")  # Show first 80 characters of the passage

# Example test with a sample query_id
query_id_to_test = "394491"
test_lookup(query_id_to_test)
