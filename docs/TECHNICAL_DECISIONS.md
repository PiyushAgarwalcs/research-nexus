# ResearchNexus — Technical Decisions

## 1. Reranker Model Architecture
**Decision:** Use `sentence_transformers.CrossEncoder` to load both pretrained and fine-tuned models.  
**Reason:** The existing code already uses CrossEncoder. The fine-tuned checkpoint at `Notebook/researchnexus-finetuning/minilm-scifact-finetuned/` is a BertForSequenceClassification model with the same architecture as `cross-encoder/ms-marco-MiniLM-L-6-v2`. CrossEncoder can load from a local directory path, making integration seamless.

## 2. Embedding Model
**Decision:** Keep OpenAI `text-embedding-3-small` for dense retrieval.  
**Reason:** Pre-existing choice. Provides good quality embeddings. Changing would invalidate existing evaluation baselines.

## 3. Vector Database
**Decision:** Keep ChromaDB for vector storage.  
**Reason:** Pre-existing choice. Lightweight, embedded, no external service needed. Suitable for the project scope.

## 4. BM25 Implementation
**Decision:** Keep LangChain's BM25Retriever with custom `bm25_tokenize`.  
**Reason:** Pre-existing. Custom tokenizer handles hyphenated terms and case normalization. Demonstrated measurable improvement in evaluation.

## 5. LLM Integration
**Decision:** Keep OpenAI gpt-4o-mini for generation and judging.  
**Reason:** Pre-existing choice. Cost-effective, good quality for RAG generation and citation verification.

## 6. Chunking Strategy
**Decision:** Default to TokenRecursive, support Markdown and Semantic.  
**Reason:** Pre-existing. TokenRecursive provides consistent chunk sizes with tiktoken-based splitting. Markdown-aware chunking preserves document structure. All three are supported and evaluated.

## 7. Frontend Framework
**Decision:** Use Streamlit for the professional frontend.  
**Reason:** Pre-existing choice. Streamlit supports multi-page apps, rich visualizations (charts, metrics, tables), file upload widgets, and is already in requirements.txt. No need to introduce a separate frontend framework for a portfolio project.

## 8. API Framework
**Decision:** Keep FastAPI for the backend API.  
**Reason:** Pre-existing. FastAPI provides automatic OpenAPI docs, Pydantic validation, async support, and is production-ready.

## 9. Configuration Management
**Decision:** Use pydantic-settings (BaseSettings) with .env file support.  
**Reason:** Pre-existing. Type-safe, environment-variable overridable, centralized configuration.

## 10. Model Checkpoint Storage
**Decision:** Keep fine-tuned model in `Notebook/researchnexus-finetuning/minilm-scifact-finetuned/` and reference it via config.  
**Reason:** The checkpoint is ~87MB (model.safetensors). Too large for typical GitHub repos. Document download/setup instructions. Add to .gitignore if not already tracked, or use Git LFS.

## 11. Evaluation Methodology
**Decision:** Report all metrics honestly including SCIDOCS performance drop.  
**Reason:** Scientific integrity. The fine-tuned model improves on SciFact but degrades on SCIDOCS, indicating domain-specific overfitting. This is a valid and honest finding.

## 12. Evidence Stance Classification
**Decision:** Use LLM-based classification with structured output for evidence stance (supporting/contradicting/neutral).  
**Reason:** Scientific evidence stance requires semantic understanding beyond keyword matching. The existing citation verification pattern (structured output via Pydantic) provides a proven approach.

## 13. RRF Score Interpretation
**Decision:** Never expose raw RRF scores as calibrated confidence values.  
**Reason:** RRF scores are relative ranking signals (~0.01 range), not probabilities. The code already notes this in comments. Cross-encoder scores are normalized via sigmoid for confidence scoring.
