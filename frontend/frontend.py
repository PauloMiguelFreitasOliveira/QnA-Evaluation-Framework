import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt

def safe_avg_precision(precisions):
    return sum(precisions) / len(precisions) if precisions else 0

# Load data
with open('qna_eval/results/evaluation_results.json') as f:
    data = json.load(f)

# Split retrieval vs generative
retrieval_entries = []
generative_entries = []
for e in data:
    if 'retrieval_method' in e and e['retrieval_method']:
        retrieval_entries.append(e)
    elif e.get('evaluation_method') == 'Generative':
        generative_entries.append(e)

# --------------------------------------------------------------------------------
# Retrieval filters
# --------------------------------------------------------------------------------
st.sidebar.markdown("### Retrieval Filters")

ret_models = st.sidebar.multiselect(
    "Models (retrieval)", 
    options=sorted({e['model_name'] for e in retrieval_entries}), 
    key="ret_models"
)
ret_methods = st.sidebar.multiselect(
    "Methods (retrieval)", 
    options=sorted({e['retrieval_method'] for e in retrieval_entries if e.get('retrieval_method')}), 
    key="ret_methods"
)
ret_datasets = st.sidebar.multiselect(
    "Datasets (retrieval)", 
    options=sorted({e['dataset_name'] for e in retrieval_entries}), 
    key="ret_datasets"
)

# Apply retrieval filters
if ret_models:
    retrieval_entries = [e for e in retrieval_entries if e['model_name'] in ret_models]
if ret_methods:
    retrieval_entries = [e for e in retrieval_entries if e.get('retrieval_method') in ret_methods]
if ret_datasets:
    retrieval_entries = [e for e in retrieval_entries if e['dataset_name'] in ret_datasets]

    
# --------------------------------------------------------------------------------
# Generative filters
# --------------------------------------------------------------------------------
st.sidebar.markdown("### Generative Filters")
gen_models = st.sidebar.multiselect(
    "Models (generative)", 
    options={e['model_name'] for e in generative_entries}, 
    key="gen_models"
)
# you could also filter by dataset or anything else
gen_datasets = st.sidebar.multiselect(
    "Datasets (generative)", 
    options={e['dataset_name'] for e in generative_entries}, 
    key="gen_datasets"
)

# Apply generative filters
if gen_models:
    generative_entries = [e for e in generative_entries if e['model_name'] in gen_models]
if gen_datasets:
    generative_entries = [e for e in generative_entries if e['dataset_name'] in gen_datasets]




st.title("📚 QnA Evaluation Dashboard")

# ─── Retrieval Section ─────────────────────────────────────────────────────────
st.header("🔍 Retrieval Evaluations")
if retrieval_entries:
    rows = []
    for entry in retrieval_entries:
        m = entry['metrics']
        rows.append({
            'model': entry['model_name'],
            'method': entry['retrieval_method'],
            'dataset': entry['dataset_name'],
            'F1': m['squad']['f1'],
            'MAP': m['retrieval'].get('map'),
            'nDCG': m['retrieval'].get('ndcg'),
            'Prec. @ K': safe_avg_precision(m['bleu']['precisions']),
        })
    df_ret = pd.DataFrame(rows)

    st.subheader("Retrieval Results Table")
    st.dataframe(df_ret, use_container_width=True)

    # F1 by model
    st.subheader("Retrieval: F1 by Model")
    fig, ax = plt.subplots(figsize=(8,4))
    df_ret.groupby('model')['F1'].mean().plot.bar(ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("F1")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

    # Retrieval metrics by method
    st.subheader("Retrieval Metrics by Method")
    agg = df_ret.groupby('method')[['MAP','nDCG','Prec. @ K']].mean().dropna(how='all')
    fig2, ax2 = plt.subplots(figsize=(8,4))
    agg.plot.bar(ax=ax2)
    ax2.set_xlabel("Method")
    ax2.set_ylabel("Score")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
else:
    st.write("No retrieval evaluations found.")

# ─── Generative Section ────────────────────────────────────────────────────────
st.header("🤖 Generative Evaluations")
if generative_entries:
    # figure out hallucination keys (PHR/THR or hallucination_rate/misinformation_rate)
    all_hkeys = sorted({
        k for e in generative_entries
        for k in e['metrics'].get('truthfulqa', {}).keys()
    })

    rows = []
    for entry in generative_entries:
        m = entry['metrics']
        base = {
            'model': entry['model_name'],
            'F1': m['squad']['f1'],
            'ROUGE-L': m['rouge']['rougeL'],
            'BLEU': m['bleu']['bleu'],
        }
        for k in all_hkeys:
            base[k] = m.get('truthfulqa', {}).get(k)
        rows.append(base)
    df_gen = pd.DataFrame(rows)

    st.subheader("Generative Results Table")
    st.dataframe(df_gen, use_container_width=True)

    # F1 by model
    st.subheader("Generative: F1 by Model")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    df_gen.groupby('model')['F1'].mean().plot.bar(ax=ax3)
    ax3.set_xlabel("Model")
    ax3.set_ylabel("F1")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig3)

    # Hallucination metrics by model
    if all_hkeys:
        st.subheader("Hallucination Metrics by Model")
        fig4, ax4 = plt.subplots(figsize=(8,4))
        df_gen.groupby('model')[all_hkeys].mean().plot.bar(ax=ax4)
        ax4.set_xlabel("Model")
        ax4.set_ylabel("Rate")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig4)

    st.subheader("Inspect Model Answer Examples")
    # build a label for each entry
    labels = [
        f"{e['model_name']} on {e['dataset_name']} @ {e['timestamp']}"
        for e in generative_entries
        if "examples" in e
    ]
    if labels:
        choice = st.selectbox("Pick an evaluation to inspect", labels)
        # find the matching entry
        sel = next(e for e in generative_entries
                   if f"{e['model_name']} on {e['dataset_name']} @ {e['timestamp']}" == choice)
        for ex in sel["examples"]:
            with st.expander(f"Q: {ex['query']}", expanded=False):
                st.write("**Ground truth:**")
                for ans in ex["ground_truth"]:
                    st.write(f"- {ans}")
                st.write("**Model’s answer:**")
                st.write(f"> {ex['prediction']}")


else:
    st.write("No generative evaluations found.")



