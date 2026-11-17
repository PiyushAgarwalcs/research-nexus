# FastAPI REST API Reference

ResearchNexus provides a production-grade FastAPI service exposing 15 REST endpoints for scientific query synthesis, dynamic reranker selection, document management, indexing, metrics monitoring, and benchmark evaluation data.

Interactive OpenAPI documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 1. System & Monitoring Endpoints

### 1.1 `GET /health`
Returns system health, active reranker, pipeline initialization status, and device info.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "pipeline_initialized": true,
  "active_reranker": "minilm-scifact-finetuned",
  "reranker_model": "finetuned",
  "device": "cpu",
  "total_requests": 142
}
```

### 1.2 `GET /metrics`
Returns real-time performance and system telemetry: total queries, cache hits, average latency, and token throughput.

**Response `200 OK`**:
```json
{
  "total_queries": 142,
  "successful_queries": 140,
  "failed_queries": 2,
  "abstention_count": 8,
  "average_latency_ms": 412.5,
  "p95_latency_ms": 780.2,
  "cache_hit_rate": 0.35,
  "active_reranker": "minilm-scifact-finetuned"
}
```

---

## 2. Model Management Endpoints

### 2.1 `GET /models`
Lists all available reranker models, active status, architecture, and benchmark summaries.

**Response `200 OK`**:
```json
{
  "active_model": "finetuned",
  "active_model_name": "minilm-scifact-finetuned",
  "models": [
    {
      "id": "finetuned",
      "name": "minilm-scifact-finetuned",
      "type": "Cross-Encoder",
      "base_architecture": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "dataset": "SciFact",
      "scifact_test_ndcg10": 0.7144,
      "scidocs_test_ndcg10": 0.1457,
      "status": "active",
      "description": "Fine-tuned on SciFact claim-evidence retrieval"
    },
    {
      "id": "pretrained",
      "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "type": "Cross-Encoder",
      "base_architecture": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "dataset": "MS-MARCO (Generic)",
      "scifact_test_ndcg10": 0.6762,
      "scidocs_test_ndcg10": 0.1634,
      "status": "available",
      "description": "Pretrained general-domain MS-MARCO reranker"
    }
  ]
}
```

### 2.2 `POST /models/select`
Dynamically switches the active cross-encoder without requiring a service restart.

**Request Body**:
```json
{
  "model": "pretrained"
}
```
*(Accepts `"pretrained"` or `"finetuned"`)*

**Response `200 OK`**:
```json
{
  "status": "success",
  "active_model": "pretrained",
  "message": "Switched active reranker to cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

---

## 3. Query & Evidence Retrieval Endpoints

### 3.1 `POST /query`
Performs hybrid search, cross-encoder reranking, stance classification, citation extraction, and answer synthesis.

**Request Body**:
```json
{
  "query": "Does 0-dimensional nanomaterials exhibit cytotoxicity?",
  "dense_weight": 0.7,
  "sparse_weight": 0.3,
  "top_k": 5,
  "reranker_model": "finetuned",
  "include_evidence_analysis": true
}
```

**Response `200 OK`**:
```json
{
  "query": "Does 0-dimensional nanomaterials exhibit cytotoxicity?",
  "answer": "Carbon-based zero-dimensional nanomaterials exhibit minimal cytotoxicity in normal human fibroblasts at concentrations below 50 µg/mL [1]. However, higher concentrations induce reactive oxygen species generation [2].",
  "confidence_score": 0.88,
  "is_fallback": false,
  "active_reranker": "minilm-scifact-finetuned",
  "citations": [
    {
      "id": 1,
      "doc_id": "scifact_abstract_104",
      "title": "Cytotoxicity evaluation of graphene quantum dots",
      "doi": "10.1039/c2nr30144c",
      "text": "Zero-dimensional carbon dots demonstrated high biocompatibility and negligible cytotoxicity...",
      "score": 0.92,
      "verified": true
    }
  ],
  "evidence_analysis": {
    "overall_consensus": "SUPPORTED",
    "consensus_confidence": 0.85,
    "support_count": 2,
    "contradict_count": 0,
    "neutral_count": 1,
    "passages": [
      {
        "passage_id": "chunk_104_0",
        "doc_id": "scifact_abstract_104",
        "title": "Cytotoxicity evaluation of graphene quantum dots",
        "stance": "SUPPORT",
        "confidence": 0.90,
        "key_quote": "negligible cytotoxicity below 50 µg/mL",
        "rationale": "Direct experimental confirmation of biocompatibility"
      }
    ]
  },
  "latency_ms": 420.3
}
```

---

## 4. Document Management Endpoints

### 4.1 `GET /documents`
Lists all documents currently indexed in the knowledge base with extracted scientific metadata.

### 4.2 `POST /documents/upload`
Uploads a scientific paper (PDF, Markdown, HTML, TXT), extracts metadata (DOI, year, authors), and triggers deduplication.

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@paper.pdf"
```

### 4.3 `DELETE /documents/{filename}`
Deletes an ingested document and its corresponding passage vectors from the index.

### 4.4 `POST /documents/index`
Forces index rebuilding or re-synchronization across dense and lexical indexes.

---

## 5. Evaluation Data Endpoints

### 5.1 `GET /evaluation/summary`
Returns summarized benchmark evaluation metrics across SciFact and SCIDOCS.

**Response `200 OK`**:
```json
{
  "scifact_val": {
    "pretrained_ndcg10": 0.6746,
    "finetuned_ndcg10": 0.7162,
    "delta_ndcg10": 0.0417,
    "p_value": 0.027
  },
  "scifact_test": {
    "pretrained_ndcg10": 0.6762,
    "finetuned_ndcg10": 0.7144,
    "delta_ndcg10": 0.0382,
    "ci_95_lower": 0.0135,
    "ci_95_upper": 0.0642,
    "p_value": 0.0006
  },
  "scidocs_test": {
    "pretrained_ndcg10": 0.1634,
    "finetuned_ndcg10": 0.1457,
    "delta_ndcg10": -0.0177,
    "ci_95_lower": -0.0230,
    "ci_95_upper": -0.0124,
    "p_value": 0.00002
  }
}
```

### 5.2 `GET /evaluation/results`
Returns comprehensive granular experiment tables, training losses, negative ablation results, and diagnostic plot paths.
