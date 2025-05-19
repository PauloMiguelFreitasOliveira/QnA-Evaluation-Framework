import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt


# 💬 Safe division helper
def safe_avg_precision(precisions):
    return sum(precisions) / len(precisions) if precisions else 0

# Load your logging file
with open('qna_eval/results/evaluation_results.json') as f:
    data = json.load(f)

# Turn it into a DataFrame for easier filtering
rows = []
for entry in data:
    metrics = entry['metrics']
    row = {
        'timestamp': entry['timestamp'],
        'model_name': entry['model_name'],
        'retrieval_method': entry['retrieval_method'],
        'dataset_name': entry['dataset_name'],
        'exact_match': metrics['squad']['exact_match'],
        'f1': metrics['squad']['f1'],
        'map': metrics['retrieval']['map'],
        'ndcg': metrics['retrieval']['ndcg'],
        'rouge1': metrics['rouge']['rouge1'],
        'rouge2': metrics['rouge']['rouge2'],
        'rougeL': metrics['rouge']['rougeL'],
        'bleu': metrics['bleu']['bleu'],
        'avg_precision': safe_avg_precision(metrics['bleu']['precisions']),
    }
    rows.append(row)

df = pd.DataFrame(rows)


# Sidebar filters
st.sidebar.header("Filters")
datasets = st.sidebar.multiselect("Select datasets:", df['dataset_name'].unique())
retrieval_methods = st.sidebar.multiselect("Select retrieval methods:", df['retrieval_method'].unique())
models = st.sidebar.multiselect("Select models:", df['model_name'].unique())

# Apply filters
if datasets:
    df = df[df['dataset_name'].isin(datasets)]
if retrieval_methods:
    df = df[df['retrieval_method'].isin(retrieval_methods)]
if models:
    df = df[df['model_name'].isin(models)]

# Main Title
st.title("📚 QnA Evaluation Dashboard")

# Display the full DataFrame
st.subheader("Evaluation Results Table")
st.dataframe(df, use_container_width=True)

# Metrics Bar Chart
st.subheader("F1 Score Comparison")

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(df['model_name'], df['f1'], color='skyblue')
ax.set_xlabel("Model Name", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("F1 Scores by Model", fontsize=16)
ax.set_xticklabels(df['model_name'], rotation=30, ha='right')  # Rotate x-labels nicely
ax.grid(axis='y', linestyle='--', alpha=0.7)

st.pyplot(fig)

# Retrieval Metrics Comparison
st.subheader("📈 Retrieval Metrics by Method")

# Group by retrieval method and average the scores
retrieval_df = df.groupby('retrieval_method')[['map', 'ndcg', 'avg_precision']].mean().reset_index()

# Plot grouped bar chart
fig2, ax2 = plt.subplots(figsize=(12, 6))

bar_width = 0.25
x = range(len(retrieval_df))

ax2.bar([i - bar_width for i in x], retrieval_df['map'], width=bar_width, label='MAP', color='mediumseagreen')
ax2.bar(x, retrieval_df['ndcg'], width=bar_width, label='nDCG', color='steelblue')
ax2.bar([i + bar_width for i in x], retrieval_df['avg_precision'], width=bar_width, label='Avg Precision', color='orange')

ax2.set_xlabel("Retrieval Method", fontsize=12)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("Retrieval Metrics by Retrieval Method", fontsize=16)
ax2.set_xticks(list(x))
ax2.set_xticklabels(retrieval_df['retrieval_method'], rotation=30, ha='right')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.6)

st.pyplot(fig2)
