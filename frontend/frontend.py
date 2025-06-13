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

PROCESS_EXPLANATIONS = {
    "Loading dataset": "Loading the evaluation dataset into memory, parsing all questions and context passages, and checking for consistency.",
    "Loading primary dataset": "Loading the evaluation dataset from memory, parsing all questions and context passages, and checking for consistency.",
    "Loading secondary dataset": "Loading the hallucination dataset from memory, parsing all questions, and checking for consistency.",
    "Retrieving top-K contexts": "For each question, the retriever model selects the top-K most relevant context passages out of the whole context pool. This step is crucial for downstream QA accuracy.",
    "Contexts retrieved": "The retriever has selected the most relevant passages for every question. These will be provided to the reader/generative model.",
    "Evaluating retrieval metrics": "Now, we calculate information retrieval metrics such as MAP (Mean Average Precision), nDCG (Normalized Discounted Cumulative Gain), and MRR (Mean Reciprocal Rank). These metrics measure how well the retriever surfaced relevant contexts.",
    "Finished retrieval metrics": "All retrieval quality metrics have been computed.",
    "Loading reader model": "The system is now loading the extractive QA (reader) model. This could be a large neural network that will generate answers to each question based on the top retrieved contexts.",
    "Extracting answers with reader": "The reader model processes each question and its retrieved contexts, generating the most likely answer span(s) for each query.",
    "Evaluating QA metrics": "SQuAD-style evaluation: Computing Exact Match and F1, as well as ROUGE and BLEU metrics, comparing the model's answers to the ground truth from the dataset.",
    "Finished all QA evaluations": "All question answering metrics have been computed.",
    "Saving results": "Writing the evaluation summary and detailed results to disk for later analysis and dashboard display.",
    "Generating primary answers": "The selected generative model is generating answers for the primary dataset's queries.",
    "Generating secondary predictions (hallucination eval)": "The model is being tested on a secondary dataset (TruthfulQA) to assess factual consistency and hallucination levels.",
    "Judging hallucinations": "A fact-checking judge model is evaluating if each generated answer is fully supported by the answer list or likely hallucinated.",
    "Preparing examples and output": "The evaluation is compiling sample questions, model answers, and ground truths for you to inspect.",
    "Preparing output": "Organizing outputs for display and saving.",
    "Evaluation Complete": "Evaluation completed successfully. Results and logs are now available.",
}
METRIC_EXPLANATIONS = {
    "Exact Match": "Percentage of answers that exactly match the ground truth.",
    "F1": "The harmonic mean of precision and recall for the predicted answers.",
    "ROUGE-L": "Longest Common Subsequence metric, measuring overlap between prediction and reference.",
    "ROUGE1": "Overlap of unigrams between prediction and ground truth.",
    "ROUGE2": "Overlap of bigrams between prediction and ground truth.",
    "ROUGEL": "Longest Common Subsequence, another variant.",
    "ROUGELSUM": "Summarization variant of ROUGE-L.",
    "BLEU": "Bilingual Evaluation Understudy; measures n-gram overlap between prediction and ground truth.",
    "MAP": "Mean Average Precision: measures the precision of the retriever at all ranks.",
    "nDCG": "Normalized Discounted Cumulative Gain: rewards higher-ranked relevant results.",
    "Prec. @ K": "Average precision of the top-K retrieved passages.",
    "PHR": "Partial Hallucination Rate: % of answers partially supported by the context.",
    "THR": "Total Hallucination Rate: % of answers completely unsupported by the context.",
}


def display_metrics_with_explanations(metrics_dict):
    for metric, value in metrics_dict.items():
        # Try to map to a friendly name
        label = metric.replace("_", " ").title()
        # Use upper-case versions as well for BLEU, ROUGE, etc
        if label.upper() in METRIC_EXPLANATIONS:
            expl = METRIC_EXPLANATIONS[label.upper()]
        else:
            expl = METRIC_EXPLANATIONS.get(label, "")
        with st.expander(f"{label}: {value}"):
            st.write(expl)


def safe_avg_precision(precisions):
    return sum(precisions) / len(precisions) if precisions else 0

def is_noisy(line):
    # Add patterns as needed
    noise_patterns = [
        "This IS expected",
        "This IS NOT expected",
        "Some weights of the model checkpoint",
        "Device set to use cpu",
        "Downloading",
        "Loading weights",
    ]
    return any(pat in line for pat in noise_patterns)

# Paths for your evaluation scripts
EVAL_SCRIPTS = {
    "Retrieval":  "retrieval_eval.py",
    "Generative": "generative_eval.py",
    "RAG":         "rag_eval.py"
}

def run_command(cmd, out_q):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        out_q.put(line)
    proc.wait()
    out_q.put(None)

st.set_page_config(layout="wide")
tabs = st.tabs(["🛠️ Run Evaluation", "🔄 Process", "📊 Results"])

# ─── Tab 1: Run Evaluation ────────────────────────────────────────────────────
with tabs[0]:
    st.header("⚙️ Execute New Evaluation")

    # 1) Tipo de avaliação
    eval_type = st.selectbox(
        "Choose type of Evaluation",
        options=[""] + list(EVAL_SCRIPTS.keys()),
        index=0
    )

    # 2) Backend (only for Generative and RAG)
    if eval_type in ("Generative", "RAG"):
        backend = st.selectbox(
            "Choose backend framework",
            options=["", "openai", "ollama", "hf"],
            index=0
        )
    else:
        backend = ""

    # sugestões de modelos já executados
    seen_retrievers = []
    seen_readers = []
    seen_generators = []
    try:
        with open("qna_eval/results/evaluation_results.json", encoding="utf-8") as f:
            prev = json.load(f)
            for e in prev:
                method = e.get("evaluation_method")
                retr = e.get("retrieval_method") or method
                if method in ("Generative", "RAG"):
                    if e.get("model_name"):
                        seen_generators.append(e.get("model_name", "UNKNOWN"))
                else:
                    if retr:
                        seen_retrievers.append(retr)
                    if e.get("model_name"):
                        seen_readers.append(e.get("model_name", "UNKNOWN"))
        seen_retrievers = sorted(set(seen_retrievers))
        seen_readers = sorted(set(seen_readers))
        seen_generators = sorted(set(seen_generators))
    except FileNotFoundError:
        pass

    # 3) Modelo — combiando pré-definidos + sugestões, com campo custom
    BASE_RETRIEVERS = ["BM25", "facebook/dpr-question_encoder-single-nq-base", "DPR"]
    BASE_READERS    = ["deepset/roberta-base-squad2", "timpal0l/mdeberta-v3-base-squad2", "ahotrod/electra_large_discriminator_squad2_512"]
    BASE_MODELS = {
        "Generative": ["gpt-4o-mini", "deepseek-llm:latest", "t5-base"],
        "RAG":         ["rag-faq", "rag-docs"]
    }

    if eval_type == "Retrieval":
        ret_opts = sorted(set(BASE_RETRIEVERS + seen_retrievers))
        ret_choice = st.selectbox(
            "Retriever model",
            options=[""] + ret_opts + ["Other…"],
            index=0
        )
        if ret_choice == "Other…":
            retriever_model = st.text_input("Insert retriever", key="custom_retriever").strip()
        else:
            retriever_model = ret_choice

        read_opts = sorted(set(BASE_READERS + seen_readers))
        read_choice = st.selectbox(
            "Reader model (extractive)",
            options=[""] + read_opts + ["Other…"],
            index=0
        )
        if read_choice == "Other…":
            reader_model = st.text_input("Insert reader", key="custom_reader").strip()
        else:
            reader_model = read_choice
        model = ""
    else:
        known_models = sorted(set(BASE_MODELS.get(eval_type, []) + seen_generators))
        model_choice = st.selectbox(
            "Choose a model",
            options=[""] + known_models + ["Other…"],
            index=0
        )
        if model_choice == "Other…":
            model = st.text_input("Insert retriever", key="custom_model").strip()
        else:
            model = model_choice
    
    # 4) Dataset — mesmo padrão
    known_datasets = [f[:-5] for f in os.listdir("datasets") if f.endswith(".json")]
    dataset_choice = st.selectbox(
        "Choose a dataset",
        options=[""] + sorted(known_datasets) + ["Other…"],
        index=0
    )
    if dataset_choice == "Other…":
        ds = st.text_input("Insert a dataset", key="custom_dataset").strip()
    else:
        ds = dataset_choice

    # 5) Número de exemplos
    limit = st.slider("Number of Entries", 1, 1000, 100)

    if eval_type == "Retrieval":
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=10, step=1)
        max_ctx = st.number_input("Max context per query", min_value=1, max_value=1000, value=100, step=1)

    if eval_type == "Retrieval":
        run_enabled = bool(eval_type and retriever_model and ds)
    else:
        run_enabled = bool(eval_type and backend and model and ds)

    # Helper to append results
    results_file = os.path.join(os.path.dirname(__file__), '..', 'qna_eval', 'results', 'evaluation_results.json')
    def append_to_results(new_entry, path=results_file):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append(new_entry)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    if st.button("▶️ Execute evaluation", disabled=not run_enabled):
        script = EVAL_SCRIPTS[eval_type]
        cmd = None
        env_vars = None

        if eval_type == "Retrieval":
            lower_ret = retriever_model.lower()
            if lower_ret in ("bm25", "dpr", "hybrid"):
                script = f"qna_eval/retrieval_files/retrieval_{lower_ret.upper()}.py"
                cmd = ["python", script]
                env_vars = os.environ.copy()
                env_vars["NUM_EXAMPLES"] = str(limit)
                # Optionally: If your script needs reader model as env, add here
                if reader_model:
                    env_vars["READER_MODEL"] = reader_model
            else:
                # For non-bm25/dpr/hybrid retrievers
                cmd = [
                    "python", script,
                    "--retriever_model", retriever_model,
                    "--reader_model", reader_model,
                    "--dataset", ds,
                    "--limit", str(limit),
                    "--top_k", str(top_k),
                    "--max_context_per_query", str(max_ctx)
                ]

        elif eval_type == "Generative":
            cmd = [
                "python", script,
                "--backend", backend,
                "--model_name", model,
                "--primary_dataset", ds,
                "--limit", str(limit),
                "--secondary_dataset", "truthfulqa",
                "--n_samples", "5"
            ]

        elif eval_type == "RAG":
            cmd = [
                "python", script,
                "--model_name", model,
                "--dataset", ds,
                "--limit", str(limit),
                "--backend", backend,
                "--secondary_dataset", "truthfulqa",
                "--n_samples", "5"
            ]

        else:
            st.error("Unknown evaluation type.")
            st.stop()

        disp_cmd = ' '.join(cmd)
        if eval_type == "Retrieval" and 'env_vars' in locals() and env_vars is not None and lower_ret in ("bm25", "dpr", "hybrid"):
            disp_cmd = f"NUM_EXAMPLES={limit} " + disp_cmd
        #st.markdown(f"**Executing:** `{ disp_cmd }`")

        # Captura em tempo real
        popen_env = env_vars if eval_type == "Retrieval" and env_vars is not None else None
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=popen_env,
            encoding="utf-8",
            errors="replace",
        )
        
        def is_progress_json(line):
            try:
                j = json.loads(line)
                return "process_stage" in j and "progress" in j and len(j) == 2
            except Exception:
                return False

        def is_final_result_json(line):
            try:
                j = json.loads(line)
                return "metrics" in j
            except Exception:
                return False

        prog = st.progress(0, text="Executing...")
        log_box = st.empty()
        logs = []
        count = 0
        current_progress = 0
        final_json = None

        st.session_state["process_log"] = []
        st.session_state["process_done"] = False

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logs.append(line)
                if is_progress_json(line):
                    progress_obj = json.loads(line)
                    st.session_state["process_log"].append(progress_obj)
                    # Update the progress bar text and percent
                    prog.progress(progress_obj["progress"], text=progress_obj["process_stage"])
                    continue
                if is_final_result_json(line):
                    # Maybe store it for use, but skip in the logs
                    final_json = line.strip()
                    continue

                # 4. Otherwise, show it in the logs if it's not noise
                if not is_noisy(line):
                    # Filtered log lines only (not progress/final)
                    cleaned_logs = [
                        l for l in logs[-20:]
                        if not (is_progress_json(l) or is_final_result_json(l))
                    ]
                    log_box.text("".join(cleaned_logs[-10:]))


        st.session_state["process_done"] = True


        ret = process.poll()
        if ret == 0:
            st.success("✅ Evaluation complete with success!")
            # Find the last single-line JSON object in the logs
            json_candidate = None
            for line in reversed(logs):
                stripped = line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        j = json.loads(stripped)
                        # Accept JSONs with metrics for all evaluation types
                        if "metrics" in j:
                            json_candidate = stripped
                            break
                    except Exception:
                        continue

            if json_candidate:
                try:
                    out = json.loads(json_candidate)
                    st.subheader("📈 Evaluation Results")
                    st.json(out, expanded=False)
                    metrics = out.get("metrics", {})

                    # --- SQuAD (F1, EM) ---
                    if "squad" in metrics:
                        st.subheader("SQuAD-style Metrics")
                        squad_metrics = metrics["squad"]
                        for k in ["exact_match", "f1"]:
                            val = squad_metrics.get(k)
                            if val is not None:
                                label = k.replace("_", " ").title()
                                with st.expander(f"{label}: {val}"):
                                    st.write(METRIC_EXPLANATIONS.get(label, ""))

                    # --- ROUGE ---
                    rouge_metrics = metrics.get("rouge")
                    if not rouge_metrics and "squad" in metrics:
                        rouge_metrics = {k: metrics["squad"][k] for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"] if k in metrics["squad"]}
                    if rouge_metrics:
                        st.subheader("ROUGE Metrics")
                        for k, val in rouge_metrics.items():
                            label = k.replace("_", "").upper()
                            with st.expander(f"{label}: {val}"):
                                st.write(METRIC_EXPLANATIONS.get(label, ""))

                    # --- BLEU ---
                    bleu_val = metrics.get("bleu")
                    if not bleu_val and "squad" in metrics:
                        bleu_val = metrics["squad"].get("bleu")
                    if bleu_val is not None:
                        st.subheader("BLEU Metric")
                        bleu_score = bleu_val.get("bleu", 0) if isinstance(bleu_val, dict) else bleu_val
                        with st.expander(f"BLEU: {bleu_score}"):
                            st.write(METRIC_EXPLANATIONS.get("BLEU", ""))

                    # --- Hallucination ---
                    if "hallucination" in metrics:
                        st.subheader("Hallucination Metrics")
                        for k, val in metrics["hallucination"].items():
                            label = k.upper()
                            with st.expander(f"{label}: {val}"):
                                st.write(METRIC_EXPLANATIONS.get(label, ""))
                    if "truthfulqa" in metrics:
                        st.subheader("TruthfulQA (Hallucination) Metrics")
                        for k, val in metrics["truthfulqa"].items():
                            label = k.upper()
                            with st.expander(f"{label}: {val}"):
                                st.write(METRIC_EXPLANATIONS.get(label, ""))

                    # --- Retrieval ---
                    if "retrieval" in metrics:
                        st.subheader("Retrieval Metrics")
                        for k, val in metrics["retrieval"].items():
                            label = k.upper()
                            with st.expander(f"{label}: {val}"):
                                st.write(METRIC_EXPLANATIONS.get(label, ""))

                    # Add to persistent results history
                    append_to_results(out)
                except Exception as e:
                    st.warning(f"Could not parse output as JSON: {e}\nRaw line: {json_candidate}")
            else:
                st.warning("No valid JSON output detected. See raw logs below.")
                st.code("".join(logs), language="text")
        else:
            st.error("❌ Error during execution:")
            st.code("".join(logs[-10:]), language="text")


# ─── Tab 2: Process ───────────────────────────────────────────────────────────

with tabs[1]:  # "🔄 Process"
    st.header("🔄 Evaluation Process")
    process_log = st.session_state.get("process_log", [])
    process_done = st.session_state.get("process_done", False)

    if not process_log:
        st.info("No evaluation in progress yet. Start a new evaluation to see process here.")
    else:
        for step in process_log:
            stage = step.get("process_stage", "Unknown Stage")
            progress = step.get("progress", 0)
            with st.expander(f"{stage} ({int(progress*100)}%)", expanded=True if not process_done else False):
                st.progress(progress)
                st.write(PROCESS_EXPLANATIONS.get(stage, ""))
        if process_done:
            st.success("✅ Evaluation finished.")

# ─── Tab 3: Results ───────────────────────────────────────────────────────────
with tabs[2]:
    st.header("📚 QnA Evaluation Dashboard")

    with st.expander("ℹ️ Click for metric definitions"):
        for k, v in METRIC_EXPLANATIONS.items():
            st.markdown(f"**{k}:** {v}")

    # Carrega resultados
    results_file = os.path.join(os.path.dirname(__file__), '..', 'qna_eval', 'results', 'evaluation_results.json')
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning("Nenhum resultado encontrado. Execute uma avaliação primeiro.")
        st.stop()

    # Separa retrieval, generative e RAG
    retrieval_entries = [e for e in data if e.get('retrieval_method')]
    generative_entries = [e for e in data if e.get('evaluation_method') == 'Generative']
    rag_entries = [e for e in data if e.get('evaluation_method') == 'RAG']

    # Sidebar Filters
    st.sidebar.markdown("### Retrieval Filters")
    ret_models = st.sidebar.multiselect(
        "Models (retrieval)",
        options=sorted({e.get('model_name', 'UNKNOWN') for e in retrieval_entries})
    )
    ret_methods = st.sidebar.multiselect(
        "Methods (retrieval)",
        options=sorted({e.get('retrieval_method', 'UNKNOWN') for e in retrieval_entries})
    )
    ret_datasets = st.sidebar.multiselect(
        "Datasets (retrieval)",
        options=sorted({e.get('dataset_name', 'UNKNOWN') for e in retrieval_entries})
    )
    if ret_models:
        retrieval_entries = [e for e in retrieval_entries if e.get('model_name', 'UNKNOWN') in ret_models]
    if ret_methods:
        retrieval_entries = [e for e in retrieval_entries if e.get('retrieval_method', 'UNKNOWN') in ret_methods]
    if ret_datasets:
        retrieval_entries = [e for e in retrieval_entries if e.get('dataset_name', 'UNKNOWN') in ret_datasets]

    st.sidebar.markdown("### Generative/RAG Filters")
    grag_entries_all = generative_entries + rag_entries
    gen_models = st.sidebar.multiselect(
        "Models (generative & rag)",
        options=sorted({e.get('model_name', 'UNKNOWN') for e in grag_entries_all})
    )
    gen_datasets = st.sidebar.multiselect(
        "Datasets (generative & rag)",
        options=sorted({e.get('dataset_name', 'UNKNOWN') for e in grag_entries_all})
    )
    gen_methods = st.sidebar.multiselect(
        "Evaluation type",
        options=sorted({e.get('evaluation_method', 'UNKNOWN') for e in grag_entries_all})
    )
    if gen_models:
        generative_entries = [e for e in generative_entries if e.get('model_name', 'UNKNOWN') in gen_models]
        rag_entries = [e for e in rag_entries if e.get('model_name', 'UNKNOWN') in gen_models]
    if gen_datasets:
        generative_entries = [e for e in generative_entries if e.get('dataset_name', 'UNKNOWN') in gen_datasets]
        rag_entries = [e for e in rag_entries if e.get('dataset_name', 'UNKNOWN') in gen_datasets]
    if gen_methods:
        if "Generative" not in gen_methods:
            generative_entries = []
        if "RAG" not in gen_methods:
            rag_entries = []

    gen_entries_graph = list(generative_entries)
    rag_entries_graph = list(rag_entries)

    table_gen = generative_entries
    table_rag = rag_entries

    # ─── Retrieval Section ───────────────────────────────────────────────────
    st.subheader("🔍 Retrieval Evaluations")
    if retrieval_entries:
        rows = []
        for entry in retrieval_entries:
            m = entry.get('metrics', {})
            rows.append({
                'Model': entry.get('model_name', 'UNKNOWN'),
                'Method': entry.get('retrieval_method', 'UNKNOWN'),
                'Dataset': entry.get('dataset_name', 'UNKNOWN'),
                'F1': m.get('squad', {}).get('f1', 0),
                'MAP': m.get('retrieval', {}).get('map', 0),
                'nDCG': m.get('retrieval', {}).get('ndcg', 0),
                'Prec. @ K': safe_avg_precision(m.get('bleu', {}).get('precisions', []))
            })
        df_ret = pd.DataFrame(rows)
        st.dataframe(df_ret, use_container_width=True)

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

    # ─── Generative & RAG Section ─────────────────────────────────────────────
    st.subheader("🤖 Generative & RAG Evaluations")
    grag_entries = generative_entries + rag_entries
    if grag_entries:
        all_hkeys = sorted({
            k for e in grag_entries
            for k in e.get('metrics', {}).get('truthfulqa', {}).keys()
        })
        rows = []
        for entry in grag_entries:
            m = entry.get('metrics', {})
            base = {
                'Model': entry.get('model_name', 'UNKNOWN'),
                'Method': entry.get('evaluation_method', 'UNKNOWN'),
                'Dataset': entry.get('dataset_name', 'UNKNOWN'),
                'Entries': entry.get('num_entries', 0),
                'F1': m.get('squad', {}).get('f1', 0),
                'ROUGE-L': m.get('rouge', {}).get('rougeL', 0),
                'BLEU': m.get('bleu', {}).get('bleu', 0)
            }
            for k in all_hkeys:
                base[k] = m.get('truthfulqa', {}).get(k, 0)
            rows.append(base)
        df_gen = pd.DataFrame(rows)
        st.dataframe(df_gen, use_container_width=True)

        col3, col4 = st.columns(2)
        if generative_entries:
            with col3:
                st.markdown("**Generative: F1 by Model**")
                dfg = pd.DataFrame({
                    'Model': [e.get('model_name', 'UNKNOWN') for e in generative_entries],
                    'F1': [e.get('metrics', {}).get('squad', {}).get('f1', 0) for e in generative_entries]
                })
                fig3, ax3 = plt.subplots(figsize=(5,3))
                dfg.groupby('Model')['F1'].mean().plot.bar(ax=ax3)
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig3)
            if all_hkeys:
                with col4:
                    st.markdown("**Generative Hallucination Rates**")
                    dfh = pd.DataFrame([
                        {'Model': e.get('model_name', 'UNKNOWN'), **{k: e.get('metrics', {}).get('truthfulqa', {}).get(k, 0) for k in all_hkeys}}
                        for e in generative_entries
                    ])
                    fig4, ax4 = plt.subplots(figsize=(5,3))
                    dfh.groupby('Model')[all_hkeys].mean().plot.bar(ax=ax4)
                    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig4)

        if rag_entries:
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("**RAG: F1 by Model**")
                dfr = pd.DataFrame({
                    'Model': [e.get('model_name', 'UNKNOWN') for e in rag_entries],
                    'F1': [e.get('metrics', {}).get('squad', {}).get('f1', 0) for e in rag_entries]
                })
                fig5, ax5 = plt.subplots(figsize=(5,3))
                dfr.groupby('Model')['F1'].mean().plot.bar(ax=ax5)
                ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig5)
            if all_hkeys:
                with col6:
                    st.markdown("**RAG Hallucination Rates**")
                    dfrh = pd.DataFrame([
                        {'Model': e.get('model_name', 'UNKNOWN'), **{k: e.get('metrics', {}).get('truthfulqa', {}).get(k, 0) for k in all_hkeys}}
                        for e in rag_entries
                    ])
                    fig6, ax6 = plt.subplots(figsize=(5,3))
                    dfrh.groupby('Model')[all_hkeys].mean().plot.bar(ax=ax6)
                    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig6)

        st.subheader("Inspect Model Answer Examples")
        labels = [
            f"{e.get('model_name', 'UNKNOWN')} on {e.get('dataset_name', 'UNKNOWN')} @ {e.get('timestamp','')}"
            for e in grag_entries if "examples" in e
        ]
        if labels:
            choice = st.selectbox("Selecione uma avaliação", ["" ] + labels, index=0)
            if choice:
                sel = next(
                    e for e in grag_entries
                    if f"{e.get('model_name', 'UNKNOWN')} on {e.get('dataset_name', 'UNKNOWN')} @ {e.get('timestamp','')}" == choice
                )
                for ex in sel.get("examples", []):
                    with st.expander(f"Q: {ex.get('query', '')}"):
                        st.markdown("**Ground truth:**")
                        for ans in ex.get("ground_truth", []):
                            st.write(f"- {ans}")
                        st.markdown("**Model’s answer:**")
                        st.write(ex.get("prediction", ""))
    else:
        st.write("No generative or RAG evaluations found.")