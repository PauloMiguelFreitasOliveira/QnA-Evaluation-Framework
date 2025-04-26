import streamlit as st
import json
import pandas as pd

# Load your logging file
with open('qna_eval/results/evaluation_results.json') as f:
    data = json.load(f)

# Turn it into a DataFrame for easier filtering
rows = []
for entry in data:
    row = {
        'timestamp': entry['timestamp'],
        'model_name': entry['model_name'],
        'retrieval_method': entry['retrieval_method'],
        'dataset_name': entry['dataset_name'],
        'exact_match': entry['metrics']['squad']['exact_match'],
        'f1': entry['metrics']['squad']['f1'],
        'map': entry['metrics']['retrieval']['map'],
        'ndcg': entry['metrics']['retrieval']['ndcg'],
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Sidebar filters
retrieval_methods = st.sidebar.multiselect("Select retrieval methods:", df['retrieval_method'].unique())
models = st.sidebar.multiselect("Select models:", df['model_name'].unique())

# Filter
if retrieval_methods:
    df = df[df['retrieval_method'].isin(retrieval_methods)]
if models:
    df = df[df['model_name'].isin(models)]

# Display
st.dataframe(df)

# Maybe plot
st.bar_chart(df.set_index('model_name')['f1'])
