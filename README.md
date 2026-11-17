# ResearchNexus — Scientific Research & Evidence Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Sentence--Transformers-yellow)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/pytest-61%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ResearchNexus** is a production-grade, evidence-grounded scientific literature retrieval, synthesis, and evidence intelligence platform. It pairs dual-index hybrid search (ChromaDB dense vectors + Okapi BM25 lexical search) with a **custom fine-tuned neural cross-encoder reranker** (`minilm-scifact-finetuned`), automated **evidence stance classification** (`SUPPORT`, `CONTRADICT`, `NEUTRAL`, `UNCERTAIN`), strict **citation provenance verification**, and an **insufficient-evidence abstention protocol**.

---

## 🌟 Key Features & Capabilities

- 🔬 **Custom Fine-Tuned Neural Reranker**: Integrated `minilm-scifact-finetuned` cross-encoder trained on scientific claim-evidence pairs, outperforming standard pretrained models by **+0.0382 nDCG@10** on held-out test data ($p < 0.001$).
- 🔀 **Hybrid Dual-Index Retrieval**: Combines semantic embeddings (`text-embedding-3-small` in ChromaDB) and exact term matching (`rank-bm25`) using Reciprocal Rank Fusion (RRF).
- ⚖️ **Scientific Evidence Stance Analysis**: Automatically classifies retrieved passages into `SUPPORT`, `CONTRADICT`, `NEUTRAL`, or `UNCERTAIN`, preventing the dangerous conflation of retrieval relevance with scientific consensus.
- 🎯 **Verifiable Citation Engine**: Generates inline academic citations (`[1]`, `[2]`) linked to verified document passages with DOI, title, and page provenance.
- 🛡️ **Zero-Hallucination Abstention**: Evaluates evidence quality and reranker confidence scores before answering; explicitly abstains with suggested alternative documents when evidence is insufficient.
- 🔄 **Dynamic Model Switching**: Hot-swap between fine-tuned and pretrained cross-encoder models at runtime via API or UI without server downtime.
- 📊 **Empirical Experiment Dashboard**: Built-in 6-page Streamlit intelligence workspace and 15 FastAPI endpoints serving live queries, metrics, and evaluation benchmarks.

---

## 🏗️ System Architecture

```
                             [ Ingested Scientific Corpus ]
                          (PDF, Markdown, HTML, TXT, SciFact)
                                          │
                                          ▼
                            [ Document Ingestion Engine ]
                       - DOI, Title, Abstract, Year, Authors
                       - SHA-256 Content Deduplication
                       - Markdown Header & Recursive Chunking
                                          │
                                          ▼
                           [ Dual-Index Representation ]
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
          [ Dense Vector Index ]                         [ BM25 Lexical Index ]
         - ChromaDB Collection                          - Rank-BM25 (k1=1.5, b=0.75)
         - text-embedding-3-small                       - Inverted Token Index
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                          [ Reciprocal Rank Fusion (RRF) ]
                             RRF_score(d) = Σ w_s / (k + rank_s(d))
                                          │
                                          ▼
                            [ Top-N Candidates Pool ]
                                    (k=20-60)
                                          │
                                          ▼
                        [ Cross-Encoder Neural Reranking ]
                           - minilm-scifact-finetuned (Fine-tuned)
                           - ms-marco-MiniLM-L-6-v2 (Pretrained)
                                          │
                                          ▼
                         [ Top-K Ranked Evidence Passages ]
                                     (k=3-5)
                                          │
                                          ▼
                        [ Scientific Evidence Analyzer ]
                      - Stance: SUPPORT | CONTRADICT | NEUTRAL
                      - Empirical Confidence Scoring
                                          │
                                          ▼
                        [ Synthesizer & Citation Engine ]
                      - Inline Bracketed Citations [1], [2]
                      - Factual Verification & Abstention Guard
```

---

## 📈 Empirical Benchmark Results

ResearchNexus underwent rigorous statistical evaluation across in-domain and transfer datasets. All experiments use **10,000 bootstrap iterations** for 95% confidence intervals and the **Wilcoxon signed-rank test** for statistical significance.

### 1. SciFact Benchmark (In-Domain Claim Verification)

| Benchmark Split | Pretrained Baseline | Fine-Tuned Model | Difference ($\Delta$) | 95% Bootstrap CI | Wilcoxon $p$-value |
|---|---:|---:|---:|---:|---:|
| **SciFact Validation** (122 queries) | 0.6746 | **0.7162** | **+0.0417** | $[+0.0047, +0.0784]$ | $p = 0.027$ |
| **SciFact Test** (300 queries) | 0.6762 | **0.7144** | **+0.0382** | $[+0.0135, +0.0642]$ | $p = 0.0006$ |

> **Finding**: Fine-tuning the cross-encoder yields a statistically significant **+5.6% relative gain** in nDCG@10 on unseen scientific claims with $p < 0.001$.

### 2. SCIDOCS Benchmark (Cross-Domain Document Recommendation)

| Benchmark Split | Pretrained Baseline | Fine-Tuned Model | Difference ($\Delta$) | 95% Bootstrap CI | Wilcoxon $p$-value |
|---|---:|---:|---:|---:|---:|
| **SCIDOCS Test** (1,000 queries) | **0.1634** | 0.1457 | **-0.0177** | $[-0.0230, -0.0124]$ | $p < 0.0001$ |

> **Critical Scientific Insight**: Fine-tuning on short biomedical claim-evidence pairs specialized the attention layers for claim verification syntax, resulting in a modest transfer decrease (-0.0177) on broad paper-to-paper recommendation. ResearchNexus transparently documents this trade-off and provides runtime model toggling.

### 3. Negative Mining Strategy Ablation (SciFact Test)

| Negative Sampling Strategy | nDCG@10 | MRR@5 | Scientific Finding |
|---|---:|---:|---|
| Pretrained Baseline | 0.6762 | 0.6489 | Zero-shot MS-MARCO weights |
| **BM25 Hard Negatives** | 0.7144 | 0.6918 | Negatives selected from lexical top results |
| **Random Negatives** | **0.7360** | **0.7188** | **+0.0216 higher than BM25 hard negatives** |

> **Why Random Negatives Won**: In scientific corpora, BM25 hard negatives often retrieve unannotated false negatives (passages exploring identical pathways or genes not marked in the sparse benchmark). Penalizing these passages introduces label noise. Random negatives provide unambiguous contrastive signals across the medical literature.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/PiyushAgarwalcs/research-nexus.git
cd research-nexus

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Set your OpenAI API key in `.env`:
```env
OPENAI_API_KEY=your_key_here
DEFAULT_RERANKER_MODEL=finetuned
```

### 3. Run Automated Test Suite

```bash
pytest -v
```
*(All 61 test cases pass across loaders, chunkers, retrieval, rerankers, evidence analysis, and API routes).*

### 4. Launch Applications

**FastAPI Service**:
```bash
uvicorn app:app --reload --port 8000
```
Interactive API documentation: `http://localhost:8000/docs`

**Streamlit Research Intelligence Dashboard**:
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` to access the 6-page interactive workspace.

**Command-Line Interface**:
```bash
python main.py --query "Does 0-dimensional nanomaterials exhibit cytotoxicity?"
```

---

## 🖥️ Streamlit Multi-Page Interface

The frontend provides 6 dedicated operational views:

1. **📊 Executive Dashboard**: System status, corpus statistics, active model banner, and real-time query metrics.
2. **🔬 Research Assistant**: Evidence-grounded Q&A interface with query confidence meters, stance badges, verified citations, and expandable source text.
3. **📚 Document Library**: Manage scientific documents, inspect extracted DOIs/authors/years, upload new papers, and view indexing status.
4. **🔍 Evidence Explorer**: Deep-dive into passage relevance scores, dense vs. BM25 ranks, cross-encoder logits, and stance justifications.
5. **⚖️ Model Comparison**: Side-by-side A/B evaluation of Pretrained vs. Fine-Tuned models on SciFact and SCIDOCS with 95% bootstrap confidence intervals.
6. **📈 Evaluation Dashboard**: Interactive loss curves, negative mining ablations, model soup weight interpolation plots, and latency benchmarks.

---

## 🔌 REST API Endpoints Overview

ResearchNexus provides 15 production endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health status, active reranker, device info |
| `/metrics` | `GET` | Query counts, latency percentiles, cache rates |
| `/models` | `GET` | List available cross-encoder models & specs |
| `/models/select` | `POST` | Dynamically switch active reranker model |
| `/query` | `POST` | Execute hybrid search, rerank, stance analysis & answer synthesis |
| `/documents` | `GET` | List indexed documents and metadata |
| `/documents/upload` | `POST` | Upload and ingest research papers |
| `/documents/{filename}` | `DELETE` | Remove document and vector passages |
| `/documents/index` | `POST` | Re-index or synchronize collections |
| `/evaluation/summary` | `GET` | Benchmark metrics (SciFact & SCIDOCS) |
| `/evaluation/results` | `GET` | Granular evaluation tables & plot assets |

See [docs/API.md](docs/API.md) for full schemas and curl examples.

---

## 📁 Repository Structure

```text
research-nexus/
├── app.py                      # FastAPI application with 15 production endpoints
├── streamlit_app.py            # Complete 6-page Streamlit intelligence platform
├── main.py                     # CLI entrypoint for querying
├── config.py                   # Pydantic centralized configuration & settings
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development & testing dependencies
├── pytest.ini                  # Pytest runner configuration
│
├── src/                        # Core application modules
│   ├── loader.py               # Scientific document loader & DOI metadata extractor
│   ├── chunker.py              # Semantic, Markdown header & token chunkers
│   ├── indexer.py              # ChromaDB vector store & Okapi BM25 indexer
│   ├── retriever.py            # Hybrid RRF & cross-encoder reranker with caching
│   ├── generator.py            # Evidence stance analyzer & citation generator
│   ├── evaluator.py            # IR evaluation harness (nDCG, MRR, Recall)
│   ├── pipeline.py             # End-to-end RAG pipeline orchestration
│   └── results_manager.py      # Results reader for training & evaluation benchmarks
│
├── Notebook/                   # Original fine-tuning experiment artifacts
│   └── researchnexus-finetuning/
│       ├── minilm-scifact-finetuned/  # Saved fine-tuned PyTorch cross-encoder
│       ├── scifact_val_results.csv    # Validation benchmark logs
│       ├── scifact_test_results.csv   # Held-out test benchmark logs
│       ├── scidocs_test_results.csv   # Generalization transfer benchmark logs
│       ├── training_losses.csv        # Epoch training loss curves
│       └── *.png                      # Diagnostic evaluation plots
│
├── tests/                      # Automated test suite (61 unit & integration tests)
│   ├── test_api.py             # REST API endpoint tests
│   ├── test_generator.py       # Evidence stance & citation parsing tests
│   ├── test_retriever.py       # RRF fusion, reranker caching & fallback tests
│   ├── test_loader.py          # Metadata extraction & deduplication tests
│   ├── test_indexer.py         # BM25 tokenization & indexing tests
│   └── test_evaluator.py       # Metric & judge evaluation tests
│
└── docs/                       # Comprehensive documentation suite
    ├── ARCHITECTURE.md         # System design, dataflow & mathematical formulations
    ├── MODEL_TRAINING.md       # PyTorch training methodology & loss dynamics
    ├── EVALUATION.md           # Benchmark reports, significance & ablation analysis
    ├── API.md                  # REST API reference and OpenAPI schema
    ├── DEVELOPMENT.md          # Local development, testing & Docker guide
    ├── IMPLEMENTATION_CHECKLIST.md  # Detailed phase-by-phase implementation tracker
    ├── TECHNICAL_DECISIONS.md  # Architectural decision records (ADRs)
    └── PROJECT_AUDIT.md        # Initial baseline architecture audit
```

---

## 📚 Complete Documentation Suite

- [System Architecture & Engineering Design](docs/ARCHITECTURE.md)
- [Neural Reranker Fine-Tuning & PyTorch Pipeline](docs/MODEL_TRAINING.md)
- [Comprehensive Evaluation & Statistical Rigor Report](docs/EVALUATION.md)
- [FastAPI REST API Reference](docs/API.md)
- [Development, Testing & Deployment Guide](docs/DEVELOPMENT.md)
- [Technical Architecture Decision Records](docs/TECHNICAL_DECISIONS.md)
- [Implementation Checklist](docs/IMPLEMENTATION_CHECKLIST.md)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
