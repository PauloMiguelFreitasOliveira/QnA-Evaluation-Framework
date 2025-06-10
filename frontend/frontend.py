# frontend.py
import streamlit as st
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import threading
import queue
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def safe_avg_precision(precisions):
    return sum(precisions) / len(precisions) if precisions else 0

# Paths for your evaluation scripts
EVAL_SCRIPTS = {
    "Retrieval":  "retrieval_eval.py",
    "Generative": "generative_eval.py",
    "RAG":         "rag_eval.py"
}

def run_command(cmd, out_q):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        out_q.put(line)
    proc.wait()
    out_q.put(None)

st.set_page_config(layout="wide")
tabs = st.tabs(["🛠️ Run Evaluation", "📊 Results"])

# ─── Tab 1: Run Evaluation ────────────────────────────────────────────────────
with tabs[0]:
    st.header("⚙️ Execute New Evaluation")

    # 1) Tipo de avaliação
    eval_type = st.selectbox(
        "Escolhe um tipo de avaliação",
        options=[""] + list(EVAL_SCRIPTS.keys()),
        index=0
    )

    # 2) Backend
    backend = st.selectbox(
        "Escolhe o backend",
        options=["", "openai", "ollama", "hf"],
        index=0
    )

    # sugestões de modelos já executados
    seen_models = []
    try:
        with open("qna_eval/results/evaluation_results.json") as f:
            prev = json.load(f)
            seen_models = sorted({e["model_name"] for e in prev})
    except FileNotFoundError:
        pass

    # 3) Modelo — combiando pré-definidos + sugestões, com campo custom
    BASE_MODELS = {
        "Retrieval":  ["bm25", "dpr", "tfidf"],
        "Generative": ["gpt-4o-mini", "deepseek-llm:latest", "t5-base"],
        "RAG":         ["rag-faq", "rag-docs"]
    }
    known_models = sorted(set(BASE_MODELS.get(eval_type, []) + seen_models))
    model_choice = st.selectbox(
        "Escolhe um modelo",
        options=[""] + known_models + ["Other…"],
        index=0
    )
    if model_choice == "Other…":
        model = st.text_input("Digite o modelo", key="custom_model").strip()
    else:
        model = model_choice

    # 4) Dataset — mesmo padrão
    known_datasets = [f[:-5] for f in os.listdir("datasets") if f.endswith(".json")]
    dataset_choice = st.selectbox(
        "Escolhe um dataset",
        options=[""] + sorted(known_datasets) + ["Other…"],
        index=0
    )
    if dataset_choice == "Other…":
        ds = st.text_input("Digite o dataset", key="custom_dataset").strip()
    else:
        ds = dataset_choice

    # 5) Número de exemplos
    limit = st.slider("Número de exemplos", 1, 1000, 100)

    run_enabled = bool(eval_type and backend and model and ds)
    if st.button("▶️ Executar avaliação", disabled=not run_enabled):
        script = EVAL_SCRIPTS[eval_type]
        cmd = [
            "python", script,
            "--backend", backend,
            "--model_name", model,
            "--primary_dataset", ds,
            "--limit", str(limit)
        ]
        if eval_type == "Generative":
            cmd += ["--secondary_dataset", "truthfulqa", "--n_samples", "5"]
        elif eval_type == "RAG":
            cmd += ["--rag_config", "default"]

        st.markdown(f"**Executando:** `{ ' '.join(cmd) }`")

        # Captura em tempo real
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        prog = st.progress(0, text="Executing...")
        log_box = st.empty()
        logs = []
        count = 0

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logs.append(line)
                count += 1
                # mostra últimas linhas
                log_box.text("".join(logs[-10:]))
                prog.progress(min(count / 50, 1.0))  # 50 linhas = 100%, estima

        ret = process.poll()
        if ret == 0:
            st.success("✅ Avaliação concluída com sucesso!")
            # filtra apenas JSON final
            json_lines = [l for l in logs if l.strip().startswith("{")]
            if json_lines:
                json_text = "\n".join(json_lines)
                try:
                    out = json.loads(json_text)
                    st.subheader("📈 Resultados da Avaliação")
                    st.json(out, expanded=False)
                    squad = out.get("metrics", {}).get("squad", {})
                    st.metric("Exact Match", squad.get("exact_match"))
                    st.metric("F1 Score", squad.get("f1"))
                except Exception:
                    st.warning("")
        else:
            st.error("❌ Erro durante execução:")
            st.code("".join(logs[-10:]))

        prog.empty()
        log_box.empty()


        

# ─── Tab 2: Results ───────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📚 QnA Evaluation Dashboard")

    # Carrega resultados
    try:
        with open('qna_eval/results/evaluation_results.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning("Nenhum resultado encontrado. Execute uma avaliação primeiro.")
        st.stop()

    # Separa retrieval & generative
    retrieval_entries = [e for e in data if e.get('retrieval_method')]
    generative_entries = [e for e in data if e.get('evaluation_method') == 'Generative']

    # Sidebar Filters
    st.sidebar.markdown("### Retrieval Filters")
    ret_models = st.sidebar.multiselect(
        "Models (retrieval)",
        options=sorted({e['model_name'] for e in retrieval_entries})
    )
    ret_methods = st.sidebar.multiselect(
        "Methods (retrieval)",
        options=sorted({e['retrieval_method'] for e in retrieval_entries})
    )
    ret_datasets = st.sidebar.multiselect(
        "Datasets (retrieval)",
        options=sorted({e['dataset_name'] for e in retrieval_entries})
    )
    if ret_models:
        retrieval_entries = [e for e in retrieval_entries if e['model_name'] in ret_models]
    if ret_methods:
        retrieval_entries = [e for e in retrieval_entries if e['retrieval_method'] in ret_methods]
    if ret_datasets:
        retrieval_entries = [e for e in retrieval_entries if e['dataset_name'] in ret_datasets]

    st.sidebar.markdown("### Generative Filters")
    gen_models = st.sidebar.multiselect(
        "Models (generative)",
        options=sorted({e['model_name'] for e in generative_entries})
    )
    gen_datasets = st.sidebar.multiselect(
        "Datasets (generative)",
        options=sorted({e['dataset_name'] for e in generative_entries})
    )
    if gen_models:
        generative_entries = [e for e in generative_entries if e['model_name'] in gen_models]
    if gen_datasets:
        generative_entries = [e for e in generative_entries if e['dataset_name'] in gen_datasets]

    # ─── Retrieval Section ───────────────────────────────────────────────────
    st.subheader("🔍 Retrieval Evaluations")
    if retrieval_entries:
        rows = []
        for entry in retrieval_entries:
            m = entry['metrics']
            rows.append({
                'Model': entry['model_name'],
                'Method': entry['retrieval_method'],
                'Dataset': entry['dataset_name'],
                'F1': m['squad']['f1'],
                'MAP': m['retrieval'].get('map'),
                'nDCG': m['retrieval'].get('ndcg'),
                'Prec. @ K': safe_avg_precision(m['bleu']['precisions'])
            })
        df_ret = pd.DataFrame(rows)
        st.dataframe(df_ret, use_container_width=True)

        # Gráficos menos largos, com rótulos em 45°
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Retrieval: F1 by Model**")
            fig, ax = plt.subplots(figsize=(5,3))
            df_ret.groupby('Model')['F1'].mean().plot.bar(ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
        with col2:
            st.markdown("**Retrieval Metrics by Method**")
            agg = df_ret.groupby('Method')[['MAP','nDCG','Prec. @ K']].mean().dropna(how='all')
            fig2, ax2 = plt.subplots(figsize=(5,3))
            agg.plot.bar(ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig2)
    else:
        st.write("No retrieval evaluations found.")

    # ─── Generative Section ───────────────────────────────────────────────────
    st.subheader("🤖 Generative Evaluations")
    if generative_entries:
        all_hkeys = sorted({
            k for e in generative_entries
            for k in e['metrics'].get('truthfulqa', {}).keys()
        })
        rows = []
        for entry in generative_entries:
            m = entry['metrics']
            base = {
                'Model': entry['model_name'],
                'F1': m['squad']['f1'],
                'ROUGE-L': m['rouge']['rougeL'],
                'BLEU': m['bleu']['bleu']
            }
            for k in all_hkeys:
                base[k] = m.get('truthfulqa', {}).get(k)
            rows.append(base)
        df_gen = pd.DataFrame(rows)
        st.dataframe(df_gen, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Generative: F1 by Model**")
            fig3, ax3 = plt.subplots(figsize=(5,3))
            df_gen.groupby('Model')['F1'].mean().plot.bar(ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig3)
        if all_hkeys:
            with col4:
                st.markdown("**Hallucination Rates by Model**")
                fig4, ax4 = plt.subplots(figsize=(5,3))
                df_gen.groupby('Model')[all_hkeys].mean().plot.bar(ax=ax4)
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig4)

        st.subheader("Inspect Model Answer Examples")
        labels = [
            f"{e['model_name']} on {e['dataset_name']} @ {e.get('timestamp','')}"
            for e in generative_entries if "examples" in e
        ]
        if labels:
            choice = st.selectbox("Selecione uma avaliação", [""] + labels, index=0)
            if choice:
                sel = next(e for e in generative_entries
                           if f"{e['model_name']} on {e['dataset_name']} @ {e.get('timestamp','')}" == choice)
                for ex in sel["examples"]:
                    with st.expander(f"Q: {ex['query']}"):
                        st.markdown("**Ground truth:**")
                        for ans in ex["ground_truth"]:
                            st.write(f"- {ans}")
                        st.markdown("**Model’s answer:**")
                        st.write(ex["prediction"])
    else:
        st.write("No generative evaluations found.")