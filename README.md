# ResearchNexus

**An evidence-grounded scientific literature retrieval and synthesis engine powered by hybrid search and a fine-tunable PyTorch reranker.**

ResearchNexus helps researchers discover relevant scientific evidence across a large collection of papers. It combines lexical search, semantic retrieval, rank fusion, neural reranking, and citation-aware answer generation to return evidence that is relevant to a research question—not just text that happens to share keywords.

> **Project status:** ResearchNexus is being developed from an existing hybrid RAG foundation. The document-ingestion, chunking, hybrid retrieval, cross-encoder reranking, answer-generation, citation-verification, API, and demo components form the starting architecture. SciFact and SCIDOCS integration, the custom PyTorch training pipeline, and the research-specific evaluation suite are planned development work unless explicitly marked as implemented in the code.

---

## Why ResearchNexus?

Scientific literature is large, technical, and constantly changing. A general-purpose language model may explain a topic, but it cannot reliably recall every relevant paper, identify the strongest supporting passage, or guarantee that a generated statement is supported by a source.

ResearchNexus addresses this with retrieval:

1. Find candidate papers and passages using both exact-term and semantic search.
2. Combine the two result lists with Reciprocal Rank Fusion (RRF).
3. Rerank candidates with a cross-encoder.
4. Give the language model a small set of ranked evidence passages.
5. Verify citations against the retrieved text and abstain when the evidence is insufficient.

The project's central machine-learning question is:

**Can a cross-encoder fine-tuned on scientific query–passage relevance data improve the quality of hybrid RAG retrieval compared with an off-the-shelf reranker?**

The answer will be established through reproducible experiments, not assumed in advance.

## Core capabilities

### Existing foundation

The current codebase provides the following building blocks:

- Multi-format document loading for PDF, HTML, Markdown, and TXT.
- Token-recursive, Markdown-header, and semantic chunking strategies.
- Dense vector indexing with ChromaDB.
- BM25 lexical indexing over the same chunks.
- Weighted Reciprocal Rank Fusion.
- Cross-encoder reranking.
- LLM answer generation using retrieved context.
- Citation checking and an insufficient-evidence response path.
- CLI, FastAPI, and Streamlit entry points.
- Retrieval evaluation and ablation scripts.

### ResearchNexus development goals

The following are the intended scientific-research extensions. Do not treat them as complete until implemented and benchmarked:

- SciFact corpus, query, and relevance-judgment integration.
- SCIDOCS evaluation integration.
- Scientific-paper-aware metadata and passage chunking.
- Training-pair construction from query–paper relevance judgments.
- BM25/dense/hybrid hard-negative mining.
- PyTorch cross-encoder fine-tuning.
- Baseline-versus-fine-tuned model comparison.
- Scientific retrieval metrics and cross-dataset evaluation.
- Research synthesis with passage-level evidence and citation traceability.

---

## System architecture

```text
                    Scientific paper corpus
                  (SciFact / additional sources)
                              |
                              v
                 Parse, normalize and chunk
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Dense vector index          BM25 index
             (ChromaDB)              (lexical search)
                 |                         |
                 +------------+------------+
                              |
                              v
                 Reciprocal Rank Fusion
                              |
                       Candidate pool
                              |
                              v
                 Cross-encoder reranker
                  (pretrained baseline)
                              |
                  [fine-tuned PyTorch
                    model is a goal]
                              |
                              v
                  Ranked evidence passages
                              |
                              v
                 LLM answer generation
                              |
                              v
                Citation / evidence checks
                              |
                              v
                Answer or abstain safely
```

### Retrieval stages

| Stage | Component | Responsibility |
|---|---|---|
| Ingestion | Document loader | Read supported research-document formats |
| Chunking | Chunker | Split documents into searchable passages |
| Indexing | ChromaDB | Store dense passage embeddings |
| Indexing | BM25 | Index lexical terms from the same passages |
| Retrieval | Dense + BM25 | Retrieve candidates using complementary signals |
| Fusion | Weighted RRF | Merge candidate rankings |
| Reranking | Cross-encoder | Score query–passage relevance |
| Generation | LLM | Synthesize an answer from retrieved evidence |
| Verification | Citation checker | Check whether cited passages support answer claims |
| Evaluation | Benchmark harness | Measure retrieval and answer quality |

---

## Datasets

ResearchNexus is designed around established scientific information-retrieval benchmarks.

### Primary dataset: SciFact

**Purpose:** Initial scientific claim/evidence retrieval experiments and development of the training-data pipeline.

- Scientific claims paired with evidence-bearing research papers.
- Labels distinguishing supporting and contradicting evidence.
- Useful for query–document relevance experiments and evidence retrieval.
- Suitable for inspecting hard negatives that discuss similar topics without providing the required evidence.

Resources:

- [SciFact on Hugging Face](https://huggingface.co/datasets/BeIR/scifact)
- [SciFact original repository](https://github.com/allenai/scifact)

### Secondary dataset: SCIDOCS

**Purpose:** Additional scientific document-retrieval evaluation and a test of generalization beyond the primary dataset.

- Scientific document retrieval benchmark.
- Includes queries, a scientific-paper corpus, and relevance judgments.
- Useful for testing whether a reranker trained or tuned on one scientific retrieval task transfers to another.

Resources:

- [SCIDOCS on Hugging Face](https://huggingface.co/datasets/BeIR/scidocs)
- [SCIDOCS original repository](https://github.com/allenai/scidocs)

### Dataset protocol

The intended experimental protocol is:

| Dataset / split | Intended use |
|---|---|
| SciFact training data | Training-pair construction and model training |
| SciFact validation data | Model selection and hyperparameter tuning |
| SciFact test data | Held-out in-domain evaluation |
| SCIDOCS | Additional retrieval/generalization evaluation |

**Prevent data leakage:** split by the benchmark's prescribed query/document partitions before generating training examples. Mine hard negatives only from the permitted training corpus. Do not use test queries or test relevance judgments for training, threshold selection, or model selection. Document any overlap between corpora and remove or report it.

SciFact labels have task-specific meanings. A supporting/contradicting evidence label should not automatically be treated as a generic relevance label without defining the training objective. ResearchNexus should keep *retrieval relevance* and *evidence stance* as distinct concepts.

---

## PyTorch reranker

The main ML contribution planned for ResearchNexus is a trainable cross-encoder.

A cross-encoder receives a query and a candidate passage together and predicts how relevant the passage is to the query. Unlike a bi-encoder, it jointly attends to the query and passage, making it useful for reranking a relatively small candidate set retrieved by BM25 and dense search.

### Planned training workflow

```text
Benchmark queries + relevance judgments
                    |
                    v
        Construct positive query/passage pairs
                    |
                    v
       Retrieve candidate negatives using
             BM25 + dense search
                    |
                    v
          Validate hard-negative labels
                    |
                    v
             Train/validation split
                    |
                    v
        Fine-tune cross-encoder in PyTorch
                    |
                    v
        Evaluate on held-out benchmark data
                    |
                    v
       Integrate selected checkpoint into RAG
```

### Candidate model experiments

The final model should be selected based on measured quality, inference latency, memory use, licensing, and compatibility with the training setup.

Potential candidates include:

- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
- [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small) with a relevance-scoring head

These are candidates, not claims that all models are already integrated.

### Training considerations

- Use a reproducible random seed and record the dataset version.
- Include hard negatives that are topically similar but not relevant to the query.
- Keep a held-out test set untouched until final evaluation.
- Track training loss and validation metrics.
- Save the best checkpoint and its tokenizer/configuration.
- Compare against the original pretrained reranker using the same candidate pool and test queries.
- Report latency and memory alongside ranking quality.

---

## Evaluation

ResearchNexus should evaluate retrieval separately from answer generation. A better answer score does not, by itself, prove that the reranker improved retrieval.

### Retrieval metrics

| Metric | What it measures |
|---|---|
| MRR@5 | How highly the first relevant result is ranked |
| Recall@5 | How many relevant results are found in the top five |
| nDCG@10 | Ranking quality with graded relevance |
| Precision@5 | Fraction of the top five results that are relevant |
| Hit@5 | Whether at least one relevant result appears in the top five |
| Evidence coverage | Whether all required evidence passages are retrieved |

Use the exact metric definitions and relevance judgments supplied by each benchmark. Do not compare numbers across datasets as if they were directly interchangeable.

### Planned experiment matrix

| Experiment | Retrieval configuration |
|---|---|
| E1 | BM25 only |
| E2 | Dense retrieval only |
| E3 | BM25 + dense retrieval + RRF |
| E4 | Hybrid retrieval + pretrained cross-encoder |
| E5 | Hybrid retrieval + fine-tuned cross-encoder |

Keep the corpus, query set, candidate depth, and evaluation protocol fixed when comparing rerankers.

### Answer-quality and system metrics

Where answer generation is enabled, also measure:

- Answer correctness against an appropriate reference.
- Faithfulness to retrieved evidence.
- Citation precision and coverage.
- Abstention quality for unanswerable questions.
- End-to-end latency and reranking latency.
- Model memory consumption.

### Results

No performance improvement is claimed in this README. Results will be added after experiments have been run and independently checked.

| Model / configuration | MRR@5 | Recall@5 | nDCG@10 | P95 latency |
|---|---:|---:|---:|---:|
| BM25 | TBD | TBD | TBD | TBD |
| Dense | TBD | TBD | TBD | TBD |
| Hybrid RRF | TBD | TBD | TBD | TBD |
| Hybrid + pretrained reranker | TBD | TBD | TBD | TBD |
| Hybrid + fine-tuned reranker | TBD | TBD | TBD | TBD |

---

## Quick start

### Requirements

- Python 3.11
- An OpenAI API key for the default embedding/generation configuration
- Internet access for initial model and dataset downloads

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys.

> The commands below describe the existing application entry points. Dataset download and indexing commands should be updated as the SciFact/SCIDOCS adapters are implemented.

### Run the test suite

```bash
pytest
```

### Run the CLI

```bash
python main.py --query "What evidence supports the research question?"
```

### Run the API

```bash
uvicorn app:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What evidence supports the research question?"}'
```

Available endpoints in the existing service:

| Endpoint | Purpose |
|---|---|
| `POST /query` | Retrieve evidence and generate an answer |
| `GET /health` | Health check |
| `GET /metrics` | Process-level request and latency metrics |

### Run the Streamlit interface

```bash
streamlit run streamlit_app.py
```

### Run with Docker

```bash
docker compose up --build
```

The existing Docker setup exposes the API on port `8000` and Streamlit on port `8501`. Check the compose configuration and environment requirements before deployment.

---

## Configuration

The existing application centralizes tunable settings in `config.py` and supports environment-variable overrides.

Example:

```bash
DENSE_WEIGHT=0.7 SPARSE_WEIGHT=0.3 FINAL_K=5 uvicorn app:app
```

Current retrieval defaults documented by the codebase:

| Setting | Default |
|---|---:|
| Dense weight | `0.7` |
| Sparse weight | `0.3` |
| Retrieval depth | `60` |
| RRF constant | `60` |
| Reranker candidate count | `20` |
| Final context count | `5` |
| Confidence threshold | `0.75` |
| Chunk size | `512` |
| Chunk overlap | `50` |

Verify these values against `config.py` before relying on them; configuration may evolve during development.

---

## Project structure

The structure below describes the current foundation plus the intended locations for new research and training modules. Create the planned files as those features are implemented.

```text
research-nexus/
├── main.py
├── app.py
├── streamlit_app.py
├── config.py
│
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── generator.py
│   ├── evaluator.py
│   └── pipeline.py
│
├── training/                  # planned PyTorch training modules
│   ├── dataset.py
│   ├── generate_pairs.py
│   ├── hard_negative_mining.py
│   ├── train_reranker.py
│   └── evaluate_reranker.py
│
├── datasets/                  # downloaded/generated data; do not commit large files
│   ├── scifact/
│   └── scidocs/
│
├── evaluation/
│   ├── benchmarks/
│   ├── reports/
│   └── results/
│
├── scripts/
│   ├── build_index.py
│   ├── eval_retrieval.py
│   ├── run_ablation.py
│   └── validate_judge.py
│
├── tests/
├── docs/
├── artifacts/                 # model checkpoints; normally stored outside Git
├── .env.example
├── requirements.txt
└── README.md
```

---

## Development roadmap

- [ ] Set up the project and run the existing tests.
- [ ] Record baseline results before modifying retrieval behavior.
- [ ] Add SciFact corpus, query, and qrels loading.
- [ ] Add scientific-paper metadata and passage-aware chunking.
- [ ] Build dense and BM25 indexes over the same passage IDs.
- [ ] Evaluate BM25, dense, and hybrid RRF retrieval.
- [ ] Construct positive and hard-negative training examples.
- [ ] Fine-tune a PyTorch cross-encoder.
- [ ] Integrate the trained checkpoint behind the existing reranker interface.
- [ ] Compare pretrained and fine-tuned rerankers on a frozen test set.
- [ ] Add SCIDOCS evaluation and document generalization results.
- [ ] Improve evidence-level citation verification.
- [ ] Publish reproducible benchmark reports and a demo.

---

## Design principles

1. **Evidence before generation:** answers should be grounded in retrieved passages.
2. **Measure retrieval independently:** evaluate ranking before attributing improvements to generation.
3. **No invented metrics:** report only results produced by reproducible experiments.
4. **Prevent leakage:** keep training, validation, and test data separate.
5. **Make models replaceable:** the reranker should be swappable without rewriting the RAG pipeline.
6. **Preserve provenance:** retain paper identifiers, passage identifiers, metadata, and source links.
7. **Abstain when evidence is insufficient:** a plausible answer is not a substitute for supporting evidence.

---

## License and attribution

Check the repository's existing `LICENSE` and retain all required copyright, license, and attribution notices when modifying or redistributing its code. Dataset and model licenses are separate: review the terms for each dataset, pretrained model, and source document before redistribution.

---

## Project objective

ResearchNexus aims to demonstrate end-to-end applied ML engineering: building a hybrid retrieval system, preparing relevance-labelled scientific data, fine-tuning a neural reranker with PyTorch, evaluating ranking improvements rigorously, and integrating the resulting model into an evidence-grounded RAG application.
