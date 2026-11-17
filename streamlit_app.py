"""ResearchNexus — Scientific Research & Evidence Intelligence Platform.

Interactive Web Application powered by:
- Hybrid Search (Chroma Dense Vector + BM25 Sparse Keyword)
- Reciprocal Rank Fusion (RRF)
- Fine-tuned Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2 on SciFact)
- Evidence-grounded Citation Verification & Stance Classification
- Model Comparison & Cross-Dataset Generalization Analytics
"""

import os
import shutil
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import RerankerModel, settings
from src.loader import multiloader
from src.pipeline import build_pipeline
from src.results_manager import results_manager

# Page Configuration
st.set_page_config(
    page_title="ResearchNexus — Evidence Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished research-grade UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-support {
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-contradict {
        background-color: #FDE8E8;
        color: #9B1C1C;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #E1EFFE;
        color: #1E429F;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-uncertain {
        background-color: #FEF08A;
        color: #713F12;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper: Cached Pipeline Loader ---

@st.cache_resource(show_spinner="Initializing ResearchNexus hybrid pipeline & models...")
def get_pipeline(reranker_choice: str):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    try:
        return build_pipeline(
            openai_api_key=api_key,
            data_dir=data_dir,
            reranker_model=reranker_choice,
        )
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None


# --- Sidebar Navigation & Global Controls ---

st.sidebar.markdown("## 🔬 ResearchNexus")
st.sidebar.caption("Evidence-Grounded Scientific Retrieval & Synthesis")

# Active Reranker Model Selector
reranker_options = {
    "Fine-tuned MiniLM (SciFact Checkpoint)": "finetuned",
    "Pretrained MiniLM (ms-marco-MiniLM-L-6-v2)": "pretrained",
}
selected_reranker_label = st.sidebar.selectbox(
    "Active Reranker Model:",
    list(reranker_options.keys()),
    index=0,
    help="Toggle between the fine-tuned checkpoint from Notebook/ and the pretrained baseline.",
)
active_reranker_id = reranker_options[selected_reranker_label]

st.sidebar.divider()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔬 Research Assistant",
        "📚 Document Library",
        "🔍 Evidence Explorer",
        "⚖️ Model Comparison",
        "📈 Evaluation Dashboard",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### System Specs")
st.sidebar.markdown(f"- **Embedding:** `{settings.embedding_model}`")
st.sidebar.markdown(f"- **Generator:** `{settings.generator_model}`")
st.sidebar.markdown(f"- **Dense / Sparse:** `{settings.dense_weight} / {settings.sparse_weight}`")
st.sidebar.markdown(f"- **Fusion Constant (k):** `{settings.rrf_k}`")
st.sidebar.markdown(f"- **Rerank Depth:** `{settings.top_n} → {settings.final_k}`")


# --- Page 1: Dashboard ---

if page == "📊 Dashboard":
    st.markdown('<div class="main-header">ResearchNexus Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Scientific Research & Evidence Intelligence Platform Overview</div>',
        unsafe_allow_html=True,
    )

    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    loader = multiloader(data_dir)
    documents = loader._document_loader(deduplicate=False) if os.path.exists(data_dir) else []

    ft_path = settings.get_finetuned_model_path()
    ft_exists = ft_path is not None and os.path.isdir(ft_path)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Indexed Documents", len(documents))
    with c2:
        st.metric("Corpus Data Directory", os.path.basename(data_dir) or "data/")
    with c3:
        st.metric("Active Reranker", "Fine-tuned" if active_reranker_id == "finetuned" else "Pretrained")
    with c4:
        st.metric("Fine-Tuned Checkpoint", "Ready (Local)" if ft_exists else "Missing")

    st.markdown("### 🏛️ Architecture Overview")
    st.markdown(
        """
        ResearchNexus combines **lexical exact-match search (BM25)** and **dense semantic search (OpenAI text-embedding-3-small)**
        via **Reciprocal Rank Fusion (RRF)**, followed by a **fine-tuned cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`
        trained on expert-annotated scientific claim-evidence pairs from SciFact).
        """
    )

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("#### Retrieval & Generation Pipeline")
        st.markdown(
            """
            1. **Multi-Format Ingestion**: PDF, Markdown, HTML, TXT with scientific metadata extraction (DOI, Authors, Year, Abstract).
            2. **Chunking Strategies**: TokenRecursive (tiktoken-aligned), MarkdownHeader, or Semantic Breakpoints.
            3. **Hybrid Search**: Chroma Vector Store (Cosine Similarity) + BM25 with hyphenated term tokenization.
            4. **Rank Fusion**: Weighted RRF score: \\(S(d) = w_d \\frac{1}{k + r_d} + w_s \\frac{1}{k + r_s}\\).
            5. **Cross-Encoder Reranking**: Full cross-attention over `[CLS] Query [SEP] Passage [SEP]`.
            6. **Evidence Stance Analysis**: Explicit classification into `SUPPORT`, `CONTRADICT`, `NEUTRAL`, `UNCERTAIN`.
            7. **Citation Verification**: Every claim is checked against cited passages; unsupported claims are flagged.
            """
        )
    with col_arch2:
        st.markdown("#### Fine-Tuning Highlights (SciFact Benchmark)")
        st.markdown(
            """
            - **Base Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 layers, 384 hidden size).
            - **SciFact In-Domain**: nDCG@10 improved from **0.6762** to **0.7144** (+0.0382, 95% CI [0.0135, 0.0642]).
            - **Cross-Domain Generalization**: Performance on SCIDOCS test dropped from **0.1634** to **0.1457**, confirming domain specialization trade-offs.
            - **Negative Mining Finding**: Random negatives outperformed BM25 hard negatives on this benchmark.
            """
        )

    st.divider()
    st.markdown("### ⚡ Quick Navigation")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("🔬 Ask Research Assistant", use_container_width=True):
            st.session_state["nav_target"] = "🔬 Research Assistant"
    with q2:
        if st.button("📚 Browse Document Library", use_container_width=True):
            st.session_state["nav_target"] = "📚 Document Library"
    with q3:
        if st.button("⚖️ Compare Model Performance", use_container_width=True):
            st.session_state["nav_target"] = "⚖️ Model Comparison"


# --- Page 2: Research Assistant ---

elif page == "🔬 Research Assistant":
    st.markdown('<div class="main-header">Scientific Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evidence-grounded answers with verifiable citations and scientific stance analysis</div>',
        unsafe_allow_html=True,
    )

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ `OPENAI_API_KEY` is not set in your `.env` file. You can still test with sample questions, but live answer generation requires an API key.")
        api_key_input = st.text_input("Enter OpenAI API Key for this session:", type="password")
        if api_key_input:
            api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input

    sample_questions = [
        "Select a sample research question or type your own...",
        "What are the main clinical findings regarding hydroxychloroquine efficacy in COVID-19 trials?",
        "When may an HTTP cache reuse a stored response without contacting the origin server?",
        "How do monoclonal antibodies target viral spike proteins in coronavirus variants?",
        "What were the primary mission objectives and landing coordinates of Apollo 11?",
    ]
    chosen_sample = st.selectbox("Sample Questions:", sample_questions)

    default_q = "" if chosen_sample == sample_questions[0] else chosen_sample
    query_text = st.text_area("Your Research Query:", value=default_q, height=90, placeholder="Enter scientific inquiry or claim to investigate...")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        submit = st.button("Synthesize Evidence", type="primary", use_container_width=True)

    with col_info:
        st.caption(f"Active Reranker: **{selected_reranker_label}**")

    if submit:
        if not query_text.strip():
            st.warning("Please provide a query first.")
        elif not api_key:
            st.error("OpenAI API Key required to run the hybrid generator.")
        else:
            with st.spinner("Executing dense + sparse retrieval, fusion, reranking, and citation verification..."):
                start_time = time.perf_counter()
                pipeline = get_pipeline(active_reranker_id)
                if pipeline is None:
                    st.error("Failed to load pipeline. Verify your documents exist in data/.")
                else:
                    if hasattr(pipeline, "retriever") and hasattr(pipeline.retriever, "switch_reranker"):
                        pipeline.retriever.switch_reranker(active_reranker_id)
                    result = pipeline.generate_robust_answer(query_text)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    status = result.get("status", "Unknown")

                    if status == "Success":
                        st.success(f"Evidence synthesized successfully in {elapsed_ms:.1f}ms")

                        # Answer Box
                        st.subheader("📝 Synthesized Answer")
                        st.write(result.get("answer", ""))

                        # Confidence Breakdown
                        conf = result.get("confidence_metrics", {})
                        if conf:
                            st.markdown("#### Confidence & Verification Signals")
                            cc1, cc2, cc3, cc4 = st.columns(4)
                            cc1.metric("Composite Score", f"{conf.get('composite_score', 0):.2f}")
                            cc2.metric("Retrieval Confidence", f"{conf.get('retrieval_confidence', 0):.2f}")
                            cc3.metric("Citation Coverage", f"{conf.get('citation_coverage', 0):.2%}")
                            cc4.metric("Completeness", f"{conf.get('answer_completeness', 0):.2f}")

                        # Citations Section
                        st.markdown("#### 🔍 Citation Verification")
                        flagged = result.get("flagged_citations", [])
                        verified = result.get("verified_citations", [])

                        if flagged:
                            st.warning(f"{len(flagged)} claim(s) flagged as NOT fully supported by cited text:")
                            for c in flagged:
                                st.markdown(f"- **Flagged Claim:** *{c.get('claim')}*  \n  **Reason:** {c.get('reasoning')}")
                        if verified:
                            st.info(f"{len(verified)} claim(s) verified as fully supported by retrieved passages.")

                        # Evidence Stance Section
                        ev_analysis = result.get("evidence_analysis", {})
                        if ev_analysis:
                            st.markdown("#### ⚖️ Scientific Evidence Stance")
                            findings = ev_analysis.get("findings", [])
                            if findings:
                                st.write(f"Consensus: {ev_analysis.get('consensus_summary', '')}")
                                sc1, sc2, sc3, sc4 = st.columns(4)
                                sc1.metric("Supporting Passages", ev_analysis.get("supporting_count", 0))
                                sc2.metric("Contradicting Passages", ev_analysis.get("contradicting_count", 0))
                                sc3.metric("Neutral / Context", ev_analysis.get("neutral_count", 0))
                                sc4.metric("Uncertain / Mixed", ev_analysis.get("uncertain_count", 0))

                                for f in findings:
                                    st_label = f.get("stance", "NEUTRAL")
                                    badge_class = f"badge-{st_label.lower()}"
                                    st.markdown(
                                        f"<span class='{badge_class}'>{st_label}</span> **Passage [{f.get('chunk_id')}]**: {f.get('key_finding')}",
                                        unsafe_allow_html=True,
                                    )
                                    if f.get("reasoning"):
                                        st.caption(f"Reasoning: {f.get('reasoning')}")

                        # Retrieved Passages Expander
                        retrieved_chunks = result.get("retrieved_chunks", [])
                        if retrieved_chunks:
                            with st.expander(f"Inspect Retrieved Evidence ({len(retrieved_chunks)} passages)"):
                                for i, chunk in enumerate(retrieved_chunks, start=1):
                                    src = chunk.metadata.get("filepath", chunk.metadata.get("filename", "Unknown"))
                                    title = chunk.metadata.get("title", "")
                                    doi = chunk.metadata.get("doi", "")
                                    st.markdown(f"**[{i}] {title or src}**" + (f" (DOI: `{doi}`)" if doi else ""))
                                    st.text(chunk.page_content)

                    else:
                        st.warning("⚠️ Insufficient Information Available")
                        st.write(result.get("reason", "Retrieved evidence does not reliably answer the query."))
                        st.info(result.get("found_context", ""))
                        suggested = result.get("suggested_documents", [])
                        if suggested:
                            st.markdown("**Suggested Documents to Consult:**")
                            for s in suggested:
                                st.markdown(f"- `{s}`")


# --- Page 3: Document Library ---

elif page == "📚 Document Library":
    st.markdown('<div class="main-header">Scientific Document Library</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Manage, inspect, upload, and index research papers with metadata preservation</div>',
        unsafe_allow_html=True,
    )

    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Document Upload Section
    st.subheader("📤 Upload Research Paper")
    uploaded_file = st.file_uploader(
        "Upload a scientific document (PDF, Markdown, TXT, HTML):",
        type=["pdf", "md", "txt", "html", "htm"],
    )

    if uploaded_file is not None:
        save_path = os.path.join(data_dir, uploaded_file.name)
        if st.button(f"Save & Index '{uploaded_file.name}'", type="primary"):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = multiloader(data_dir)
            doc = loader.load_file(save_path)
            if doc:
                st.success(f"Successfully uploaded and parsed '{uploaded_file.name}'!")
                meta = doc.metadata
                st.json({
                    "title": meta.get("title"),
                    "authors": meta.get("authors"),
                    "year": meta.get("year"),
                    "doi": meta.get("doi"),
                    "page_count": meta.get("page_count"),
                    "doc_id": meta.get("doc_id"),
                })
                st.cache_resource.clear()
            else:
                st.error("Failed to extract content from uploaded file.")

    st.divider()

    # Document Listing
    st.subheader("📑 Indexed Research Papers")
    loader = multiloader(data_dir)
    documents = loader._document_loader(deduplicate=False)

    if not documents:
        st.info("No documents currently in corpus. Upload documents above or run `python scripts/fetch_corpus.py`.")
    else:
        doc_rows = []
        for d in documents:
            m = d.metadata
            doc_rows.append({
                "Filename": m.get("filename", "unknown"),
                "Title": m.get("title") or m.get("filename"),
                "Year": m.get("year") or "N/A",
                "DOI": m.get("doi") or "N/A",
                "Pages": m.get("page_count", 1),
                "Size (KB)": round(m.get("file_size", 0) / 1024, 1),
                "Content Hash": m.get("content_hash", "N/A"),
            })

        df_docs = pd.DataFrame(doc_rows)

        # Search filter
        search_kw = st.text_input("Filter documents by title, keyword, or DOI:")
        if search_kw:
            df_docs = df_docs[
                df_docs["Title"].str.contains(search_kw, case=False, na=False)
                | df_docs["DOI"].str.contains(search_kw, case=False, na=False)
                | df_docs["Filename"].str.contains(search_kw, case=False, na=False)
            ]

        st.dataframe(df_docs, use_container_width=True)

        # Document preview & management
        selected_doc_name = st.selectbox("Select document to inspect or manage:", [r["Filename"] for r in doc_rows])
        if selected_doc_name:
            target_doc = next((d for d in documents if d.metadata.get("filename") == selected_doc_name), None)
            if target_doc:
                c_del, c_meta = st.columns([1, 3])
                with c_del:
                    if st.button(f"🗑️ Delete '{selected_doc_name}'"):
                        target_path = os.path.join(data_dir, selected_doc_name)
                        if os.path.exists(target_path):
                            os.remove(target_path)
                            st.success(f"Deleted '{selected_doc_name}'")
                            st.cache_resource.clear()
                            st.rerun()
                with c_meta:
                    st.markdown(f"**Title:** {target_doc.metadata.get('title')}")
                    if target_doc.metadata.get("doi"):
                        st.markdown(f"**DOI:** `{target_doc.metadata.get('doi')}`")
                    if target_doc.metadata.get("abstract"):
                        st.markdown(f"**Abstract:** *{target_doc.metadata.get('abstract')}*")

                with st.expander("Document Text Preview"):
                    st.text(target_doc.page_content[:2000] + ("..." if len(target_doc.page_content) > 2000 else ""))


# --- Page 4: Evidence Explorer ---

elif page == "🔍 Evidence Explorer":
    st.markdown('<div class="main-header">Evidence Explorer & Stance Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect retrieval rankings, Reciprocal Rank Fusion scores, and cross-encoder relevance</div>',
        unsafe_allow_html=True,
    )

    ev_query = st.text_input("Enter research claim or question to inspect evidence:", value="Hydroxychloroquine reduces mortality in COVID-19 patients.")
    col_k, col_mode = st.columns(2)
    with col_k:
        top_k_select = st.slider("Passages to retrieve:", min_value=2, max_value=15, value=5)
    with col_mode:
        retrieval_mode = st.selectbox("Retrieval Strategy:", ["Hybrid (Dense + Sparse)", "Dense Only", "Sparse Only"])

    if st.button("Inspect Evidence Ranks", type="primary"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("OPENAI_API_KEY needed for live vector similarity search.")
        else:
            with st.spinner("Retrieving candidate passages..."):
                pipeline = get_pipeline(active_reranker_id)
                if pipeline and hasattr(pipeline, "retriever"):
                    retriever = pipeline.retriever
                    if retrieval_mode == "Dense Only":
                        retriever.dense_weight = 1.0
                        retriever.sparse_weight = 0.0
                    elif retrieval_mode == "Sparse Only":
                        retriever.dense_weight = 0.0
                        retriever.sparse_weight = 1.0
                    else:
                        retriever.dense_weight = settings.dense_weight
                        retriever.sparse_weight = settings.sparse_weight

                    ranked_candidates = retriever.get_relevant_documents(ev_query)[:top_k_select]
                    if not ranked_candidates:
                        st.info("No documents matched the query.")
                    else:
                        st.markdown(f"### Retrieved Top {len(ranked_candidates)} Passages")
                        for rank, (chunk, score) in enumerate(ranked_candidates, start=1):
                            title = chunk.metadata.get("title", chunk.metadata.get("filename", "Document"))
                            doi = chunk.metadata.get("doi", "")
                            st.markdown(f"#### Rank #{rank} — Relevance Score: `{score:.4f}`")
                            st.caption(f"Source: **{title}**" + (f" | DOI: `{doi}`" if doi else ""))
                            st.info(chunk.page_content)


# --- Page 5: Model Comparison ---

elif page == "⚖️ Model Comparison":
    st.markdown('<div class="main-header">Model Comparison: Pretrained vs Fine-Tuned</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Direct evaluation benchmarks: Pretrained MiniLM vs Fine-tuned MiniLM (SciFact)</div>',
        unsafe_allow_html=True,
    )

    comp_df = results_manager.get_model_comparison_table()
    stats_df = results_manager.get_statistical_significance()

    st.markdown("### 📊 Benchmark Performance Comparison")

    tab_scifact_val, tab_scifact_test, tab_scidocs = st.tabs([
        "SciFact Validation (In-Domain)",
        "SciFact Test (In-Domain)",
        "SCIDOCS Test (Cross-Domain)",
    ])

    with tab_scifact_val:
        st.markdown("#### SciFact Validation Split (122 Queries)")
        scifact_val_data = {
            "Metric": ["MRR@5", "Recall@5", "nDCG@10", "Precision@5"],
            "Pretrained MiniLM": [0.6350, 0.7637, 0.6746, 0.1656],
            "Fine-tuned MiniLM": [0.6663, 0.7964, 0.7162, 0.1738],
            "Delta": ["+0.0313", "+0.0328", "+0.0417*", "+0.0082"],
            "Verdict": ["Neutral", "Neutral", "Better (p=0.0425)", "Neutral"],
        }
        st.dataframe(pd.DataFrame(scifact_val_data), use_container_width=True)
        st.caption("*Statistically significant improvement on nDCG@10 (95% CI: [+0.0047, +0.0784]).")

    with tab_scifact_test:
        st.markdown("#### SciFact Test Split (300 Queries)")
        scifact_test_data = {
            "Metric": ["MRR@5", "Recall@5", "nDCG@10", "Precision@5"],
            "Pretrained MiniLM": [0.6421, 0.7352, 0.6762, 0.1627],
            "Fine-tuned MiniLM": [0.6863, 0.7728, 0.7144, 0.1713],
            "Delta": ["+0.0442*", "+0.0377*", "+0.0382*", "+0.0087"],
            "Verdict": ["Better (p=0.0046)", "Better (p=0.0307)", "Better (p=0.0054)", "Neutral"],
        }
        st.dataframe(pd.DataFrame(scifact_test_data), use_container_width=True)
        st.caption("*Significant gains on MRR@5, Recall@5, and nDCG@10.")

    with tab_scidocs:
        st.markdown("#### SCIDOCS Test Split (1000 Queries) — Cross-Dataset Generalization")
        scidocs_data = {
            "Metric": ["MRR@5", "Recall@5", "nDCG@10", "Precision@5"],
            "Pretrained MiniLM": [0.2795, 0.1212, 0.1634, 0.1196],
            "Fine-tuned MiniLM": [0.2498, 0.1070, 0.1457, 0.1056],
            "Delta": ["-0.0297*", "-0.0142*", "-0.0177*", "-0.0140*"],
            "Verdict": ["Worse (p<1e-5)", "Worse (p<1e-6)", "Worse (p<1e-10)", "Worse (p<1e-6)"],
        }
        st.dataframe(pd.DataFrame(scidocs_data), use_container_width=True)
        st.warning(
            "⚠️ **Scientific Honesty Note:** Fine-tuning on SciFact reduced performance on SCIDOCS. "
            "This demonstrates domain-specific specialization (claim verification vs document recommendation) "
            "and cross-dataset generalization limitations."
        )

    st.divider()

    # Paired Bootstrap CI Table
    st.markdown("### 🎯 Paired Bootstrap 95% Confidence Intervals (nDCG@10)")
    ci_table = [
        {"Dataset": "SciFact Validation", "Pretrained": 0.6746, "Fine-tuned": 0.7162, "Improvement": "+0.0417", "95% Confidence Interval": "[+0.0047, +0.0784]", "Significant": "Yes"},
        {"Dataset": "SciFact Test", "Pretrained": 0.6762, "Fine-tuned": 0.7144, "Improvement": "+0.0382", "95% Confidence Interval": "[+0.0135, +0.0642]", "Significant": "Yes"},
        {"Dataset": "SCIDOCS Test", "Pretrained": 0.1634, "Fine-tuned": 0.1457, "Improvement": "-0.0177", "95% Confidence Interval": "[-0.0230, -0.0124]", "Significant": "Yes (Degradation)"},
    ]
    st.dataframe(pd.DataFrame(ci_table), use_container_width=True)

    # Diagnostic Plots
    st.markdown("### 📈 Visual Benchmark Plots")
    plots = results_manager.get_available_plots()
    if plots:
        plot_names = [p["title"] for p in plots]
        selected_plot = st.selectbox("Select diagnostic plot to view:", plot_names)
        target_plot = next((p for p in plots if p["title"] == selected_plot), None)
        if target_plot and os.path.exists(target_plot["path"]):
            st.image(target_plot["path"], caption=target_plot["title"], use_container_width=True)


# --- Page 6: Evaluation Dashboard ---

elif page == "📈 Evaluation Dashboard":
    st.markdown('<div class="main-header">Evaluation & Training Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Loss curves, negative sampling ablation, and weight interpolation analysis</div>',
        unsafe_allow_html=True,
    )

    t_loss, t_ablation, t_soup = st.tabs([
        "Training Loss & Convergence",
        "Negative Mining Ablation",
        "Weight Interpolation (Model Soup)",
    ])

    with t_loss:
        st.markdown("#### Cross-Encoder Training & Validation Loss")
        loss_history = results_manager.get_training_history()
        if loss_history:
            df_loss = pd.DataFrame(loss_history)
            st.dataframe(df_loss, use_container_width=True)

            chart_data = df_loss.set_index("epoch")[["train_loss", "val_loss"]]
            st.line_chart(chart_data)
            st.caption("Epoch 3 reached lowest validation loss: 0.4467 (Early stopping checkpoint saved).")

    with t_ablation:
        st.markdown("#### Hard Negatives vs Random Negatives Ablation (nDCG@10)")
        ablation_data = [
            {"Dataset": "SciFact Validation", "Pretrained": 0.6746, "Random Negatives": 0.7227, "BM25 Hard Negatives": 0.7162},
            {"Dataset": "SciFact Test", "Pretrained": 0.6762, "Random Negatives": 0.7360, "BM25 Hard Negatives": 0.7144},
            {"Dataset": "SCIDOCS Test", "Pretrained": 0.1634, "Random Negatives": 0.1621, "BM25 Hard Negatives": 0.1457},
        ]
        st.dataframe(pd.DataFrame(ablation_data), use_container_width=True)
        st.info(
            "💡 **Finding:** BM25 hard negatives did NOT outperform random negatives. "
            "Random negatives produced stronger discriminative signal (+0.7360 on SciFact test vs 0.7144 with hard negatives), "
            "likely because lexical overlap in hard negatives introduced false negatives."
        )

    with t_soup:
        st.markdown("#### Weight Interpolation Tradeoff: Pretrained vs Fine-Tuned (α)")
        interp_df = results_manager.get_weight_interpolation()
        if not interp_df.empty:
            st.dataframe(interp_df, use_container_width=True)
            st.caption(
                "Interpolating model weights: \\(W_{\\text{interp}} = (1-\\alpha) W_{\\text{pre}} + \\alpha W_{\\text{ft}}\\). "
                "At \\(\\alpha \\approx 0.5 - 0.75\\), in-domain SciFact gains are preserved while out-of-domain SCIDOCS performance drops less."
            )
