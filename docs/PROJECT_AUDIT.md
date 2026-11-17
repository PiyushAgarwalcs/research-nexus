# ResearchNexus — Project Audit

**Audit date:** 2026-11-17

## 1. Existing Architecture

```
User Query
  │
  ▼
Document Loading (src/loader.py) ─── PDF, HTML, Markdown, TXT
  │
  ▼
Chunking (src/chunker.py) ─── TokenRecursive / Markdown / Semantic
  │
  ▼
Indexing (src/indexer.py) ─── ChromaDB (dense) + BM25Retriever (sparse)
  │
  ▼
Hybrid Retrieval (src/retriever.py) ─── Dense + Sparse → RRF Fusion → Cross-Encoder Reranking
  │
  ▼
RAG Generation (src/generator.py) ─── Citation-verified answer with confidence scoring
  │
  ▼
API Layer (app.py / streamlit_app.py) ─── FastAPI + Streamlit
```

## 2. Existing Features

### Document Loading (`src/loader.py`)
- Supports PDF (pymupdf), HTML (BeautifulSoup), Markdown, TXT
- Recursive directory traversal with sorted order for reproducibility
- Text normalization (whitespace, newlines)
- Basic metadata: filepath, extension, filename

### Chunking (`src/chunker.py`)
- TokenRecursive: tiktoken-based with configurable size/overlap
- Markdown: header-aware splitting (H1/H2/H3)
- Semantic: embedding-based with percentile breakpoints
- All strategies attach chunk_strategy metadata

### Indexing (`src/indexer.py`)
- ChromaDB for dense vector storage with OpenAI embeddings
- BM25Retriever with custom tokenizer (lowercase, hyphenated terms)
- Metadata enrichment: chunk_index, section_heading, character_count
- Reset collection before rebuilding to prevent duplicates

### Retrieval (`src/retriever.py`)
- Hybrid retrieval: Dense + BM25 with configurable weights
- Reciprocal Rank Fusion (1-based ranks, configurable rrf_k)
- Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Duplicate handling in fusion (seen set per retriever)
- Reranker can be disabled (use_reranker=False) or injected (pre-loaded instance)
- Returns (doc, score) pairs preserving cross-encoder scores

### Generation (`src/generator.py`)
- Citation-verified answer generation with bracketed references [1], [2]
- Citation parsing supports consecutive citations [1][2] and comma-separated [1, 2]
- LLM-based citation verification per claim
- Confidence scoring: retrieval (sigmoid-normalized CE scores) × 0.3 + citation coverage × 0.4 + completeness × 0.3
- Insufficient information response with suggested documents
- Cross-encoder score normalization via sigmoid

### Evaluation (`src/evaluator.py`)
- Synthetic dataset generation (Lookup, Multi-Hop, Unanswerable, Ambiguous)
- RAG evaluation: correctness, faithfulness, retrieval relevance, citation accuracy
- Objective retrieval metrics: evidence_ranks, retrieval_scores (hit@1, hit@k, full@k, recall, MRR)
- LLM-judged evaluation with structured output

### API (`app.py`)
- FastAPI with lifespan management (pipeline built once at startup)
- Endpoints: POST /query, GET /health, GET /metrics
- Token counting and cost estimation (tiktoken)
- In-memory metrics store (latency, cost)

### Frontend (`streamlit_app.py`)
- Basic single-page Streamlit demo
- Query input → answer display with confidence metrics
- Citation flagging display
- Retrieved context expandable view

### Configuration (`config.py`)
- Pydantic BaseSettings with .env support
- Retrieval params: dense_weight, sparse_weight, retrieval_depth, rrf_k, top_n, final_k
- Model params: generator_model, judge_model, embedding_model
- Generation params: confidence_threshold, temperatures

### Scripts
- `build_index.py`: CLI index builder
- `eval_retrieval.py`: Objective retrieval evaluation (no LLM needed for scoring)
- `fetch_corpus.py`: Downloads IETF RFC documents
- `run_ablation.py`: Ablation study runner (retrieval mode, RRF sweep, reranker on/off)
- `validate_judge.py`: LLM judge validation

### Tests (6 files)
- `test_retriever.py`: 10 tests — fusion, reranking, duplicate handling, reranker injection
- `test_generator.py`: Citation parsing, verification, confidence scoring
- `test_evaluator.py`: Evidence ranks, retrieval scores, split counts
- `test_indexer.py`: Metadata preparation, BM25 tokenization
- `test_loader.py`: File loading, text normalization
- `test_fetch_corpus.py`: Corpus fetcher mocking

## 3. Technology Stack
- **Language:** Python 3.11
- **Vector DB:** ChromaDB
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** OpenAI gpt-4o-mini
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)
- **Framework:** LangChain (core, community, openai, chroma, text-splitters, experimental)
- **API:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **PDF:** PyMuPDF
- **HTML:** BeautifulSoup4
- **BM25:** rank-bm25 (via LangChain BM25Retriever)
- **Config:** pydantic-settings
- **Testing:** pytest
- **CI:** GitHub Actions
- **Container:** Docker + docker-compose

## 4. Fine-Tuned Model Artifacts (Notebook/)

### Location: `Notebook/researchnexus-finetuning/minilm-scifact-finetuned/`
- `config.json` — BertForSequenceClassification, 6 layers, 384 hidden, 12 heads
- `model.safetensors` — 87MB checkpoint
- `tokenizer.json` — BertTokenizer vocabulary
- `tokenizer_config.json` — tokenizer settings
- `training_config.json` — hyperparameters (3 epochs, lr=2e-5, batch=16, fp16, seed=42)
- `training_history.csv` — per-epoch train/val loss
- `evaluation_results.json` — comprehensive metrics across SciFact (val/test), SCIDOCS (test)

### Training Data: `Notebook/researchnexus-finetuning/datasets/`
- `train_pairs.csv` — 3136 training pairs from 687 queries
- `validation_pairs.csv` — 540 validation pairs from 122 queries

### Results: `Notebook/researchnexus-finetuning/results/`
- `final_comparison.csv` — all models × all datasets
- `statistical_significance.csv` — paired bootstrap CI + Wilcoxon
- `negative_ablation.csv` — random vs BM25 hard negatives
- `weight_interpolation.csv` — model weight interpolation (alpha 0→1)
- `training_history.csv`, `baseline_metrics.csv`, `finetuned_metrics.csv`, `per_query_win_loss.csv`

### Plots: `Notebook/researchnexus-finetuning/plots/`
- training_loss.png, model_comparison.png, final_model_comparison.png
- improvement_with_ci.png, negative_ablation.png, per_query_analysis.png
- interpolation_tradeoff.png, eda_distributions.png

### Notebook: `Notebook/researchnexus.ipynb` — 748KB, complete training experiment

## 5. Missing Functionality
1. **No fine-tuned model integration** — retriever hardcodes pretrained model name
2. **No model selection mechanism** — cannot switch between pretrained/fine-tuned
3. **No scientific metadata extraction** — no title, authors, DOI, year parsing
4. **No evidence stance classification** — all evidence treated equally
5. **No document management UI** — no upload, list, delete, search
6. **No model comparison dashboard** — evaluation results not surfaced in UI
7. **No evaluation results dashboard** — training curves, metrics not displayed
8. **No .env.example** — setup instructions incomplete
9. **Basic Streamlit frontend** — single page, no multi-section layout
10. **Limited API endpoints** — no document management, no model selection, no evaluation

## 6. Bugs and Technical Limitations
- **Model checkpoint ~87MB** — too large for standard GitHub push without LFS
- **OpenAI dependency** — both embeddings and generation require OPENAI_API_KEY
- **No offline mode** — cannot run without network for embedding/generation
- **Index rebuild on every startup** — pipeline.build_pipeline rebuilds from scratch
- **No persistent document management** — documents must be in data/ directory

## 7. Integration Requirements
- Load fine-tuned model from local path using CrossEncoder
- Add model selection to config (pretrained vs fine-tuned)
- Surface evaluation results from Notebook CSV/JSON files
- Build multi-page Streamlit app
- Expand FastAPI with document management and model selection endpoints

## 8. Recommended Implementation Plan
1. **Model Integration** (highest impact, foundation for model comparison)
2. **Frontend Enhancement** (most visible improvement, showcases all features)
3. **Evidence Analysis** (differentiating feature for scientific research)
4. **API Expansion** (supports frontend functionality)
5. **Documentation** (project presentation quality)
6. **Testing** (reliability and correctness verification)
