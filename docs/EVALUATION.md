# Comprehensive Evaluation & Statistical Rigor Report

This document reports the scientific evaluation results, statistical significance tests, cross-dataset transfer benchmarks, negative ablation experiments, and latency profiling for **ResearchNexus**.

---

## 1. Evaluation Methodology & Metrics

Retrieval quality is measured independently from answer generation. The following standard Information Retrieval (IR) metrics are evaluated:

| Metric | Formal Definition | Scientific Interpretation |
|---|---|---|
| **nDCG@10** | $\text{DCG}@10 / \text{IDCG}@10$ where $\text{DCG}@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}$ | Primary metric. Evaluates graded ranking quality, rewarding top placements of highly relevant evidence. |
| **MRR@5** | $\frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$ | Mean Reciprocal Rank. Measures how quickly the first relevant piece of evidence is surfaced. |
| **Recall@5** | $\frac{|\text{Retrieved@5} \cap \text{Relevant}|}{|\text{Relevant}|}$ | Measures the proportion of all relevant evidence documents retrieved in top 5. |
| **Precision@5** | $\frac{|\text{Retrieved@5} \cap \text{Relevant}|}{5}$ | Fraction of top-5 results that are relevant. |
| **Hit@5** | $\mathbb{I}(\text{Retrieved@5} \cap \text{Relevant} \neq \emptyset)$ | Binary indicator of whether at least one relevant passage was found in top 5. |

---

## 2. Statistical Significance Protocol

To avoid reporting spurious gains, ResearchNexus employs:
1. **Paired Wilcoxon Signed-Rank Test**: A non-parametric test comparing query-level metric distributions ($H_0$: difference median is zero).
2. **Empirical Bootstrap Confidence Intervals**: 10,000 resamples with replacement to compute 95% confidence intervals ($\alpha = 0.05$) for the mean metric difference:
   $$\Delta \text{Metric} = \text{Metric}_{\text{finetuned}} - \text{Metric}_{\text{pretrained}}$$

---

## 3. Main Benchmark Results

### 3.1 SciFact Validation Set (122 Queries)

| Model / Configuration | nDCG@10 | MRR@5 | Recall@5 | Hit@5 |
|---|---:|---:|---:|---:|
| Pretrained Baseline (`ms-marco-MiniLM-L-6-v2`) | 0.6746 | 0.6512 | 0.7230 | 0.8115 |
| **Fine-Tuned Checkpoint (`minilm-scifact-finetuned`)** | **0.7162** | **0.6934** | **0.7681** | **0.8607** |
| **Difference ($\Delta$)** | **+0.0417** | **+0.0422** | **+0.0451** | **+0.0492** |
| **95% Bootstrap Confidence Interval** | **[+0.0047, +0.0784]** | [+0.0031, +0.0812] | [+0.0058, +0.0844] | — |
| **Wilcoxon $p$-value** | **$p = 0.027$** ($p < 0.05$) | $p = 0.034$ | $p = 0.021$ | — |

*Conclusion*: Statistically significant improvement on validation claims.

---

### 3.2 SciFact Test Set (300 Queries, Held-out)

| Model / Configuration | nDCG@10 | MRR@5 | Recall@5 | Hit@5 |
|---|---:|---:|---:|---:|
| Pretrained Baseline (`ms-marco-MiniLM-L-6-v2`) | 0.6762 | 0.6489 | 0.7180 | 0.8033 |
| **Fine-Tuned Checkpoint (`minilm-scifact-finetuned`)** | **0.7144** | **0.6918** | **0.7592** | **0.8500** |
| **Difference ($\Delta$)** | **+0.0382** | **+0.0429** | **+0.0412** | **+0.0467** |
| **95% Bootstrap Confidence Interval** | **[+0.0135, +0.0642]** | [+0.0151, +0.0706] | [+0.0142, +0.0682] | — |
| **Wilcoxon $p$-value** | **$p = 0.0006$** ($p < 0.001$) | $p = 0.0011$ | $p = 0.0018$ | — |

*Conclusion*: High statistical significance ($p < 0.001$) confirming that the fine-tuned reranker outperforms the baseline on unseen scientific claims.

---

## 4. Cross-Dataset Transfer & Generalization: SCIDOCS

A key question in applied ML is whether fine-tuning on a specific task/dataset transfers to broader scientific retrieval. We evaluated both models on the held-out **SCIDOCS benchmark (1,000 queries)**.

### 4.1 SCIDOCS Test Results (1,000 Queries)

| Model / Configuration | nDCG@10 | MRR@5 | Recall@5 |
|---|---:|---:|---:|
| Pretrained Baseline (`ms-marco-MiniLM-L-6-v2`) | **0.1634** | **0.1582** | **0.1912** |
| Fine-Tuned Checkpoint (`minilm-scifact-finetuned`) | 0.1457 | 0.1410 | 0.1738 |
| **Difference ($\Delta$)** | **-0.0177** | **-0.0172** | **-0.0174** |
| **95% Bootstrap Confidence Interval** | **[-0.0230, -0.0124]** | [-0.0226, -0.0118] | [-0.0232, -0.0116] |
| **Wilcoxon $p$-value** | **$p < 0.0001$** | $p < 0.0001$ | $p < 0.0001$ |

### 4.2 Critical Analysis of the SCIDOCS Performance Drop
Why does fine-tuning improve SciFact (+0.0382) while decreasing SCIDOCS (-0.0177)?
1. **Task Formulation Discrepancy**:
   - *SciFact* is a **claim verification** task: short factual claims (e.g., "0-dimensional nanomaterials lack cytotoxicity") paired with sentence/abstract-level evidence passages.
   - *SCIDOCS* is a **paper-to-paper recommendation / co-citation** task: long paper titles and abstract queries seeking conceptually related academic works across computer science, materials, and physics.
2. **Domain Overfitting / Narrowing**:
   - The fine-tuned cross-encoder learned strong attention weights for biomedical claim-evidence syntax, which penalizes broader citation relevance patterns in SCIDOCS.
3. **Engineering Takeaway**:
   - In production, one model does not fit all scientific tasks. ResearchNexus's modular architecture allows administrators to toggle between the general-purpose baseline and the claim-specific fine-tuned model via API (`POST /models/select`) or Streamlit UI.

---

## 5. Ablation Studies

### 5.1 Negative Mining Strategy Ablation (SciFact Test)

| Negative Mining Strategy | nDCG@10 | MRR@5 | Description |
|---|---:|---:|---|
| Pretrained Baseline | 0.6762 | 0.6489 | Zero-shot on SciFact |
| BM25 Hard Negatives | 0.7144 | 0.6918 | Negatives selected from top lexical retrieval |
| **Random Negatives** | **0.7360** | **0.7188** | Negatives sampled uniformly from the corpus |

**Finding**: Random negative mining yielded +0.0216 higher nDCG@10 than BM25 hard negatives. This occurs because BM25 hard negatives in biomedical corpora frequently contain unannotated false negatives (passages addressing the same pathway/disease that were not annotated in the dataset). Random negatives provide clean, unconfounded contrastive signals.

### 5.2 Retrieval Pipeline Stage Ablation

| Stage Configuration | SciFact Test nDCG@10 | Latency (P50) |
|---|---:|---:|
| Lexical BM25 Only | 0.5841 | 8 ms |
| Dense Vector Retrieval Only (`text-embedding-3-small`) | 0.6215 | 32 ms |
| Hybrid RRF (BM25 + Dense) | 0.6582 | 41 ms |
| Hybrid RRF + Pretrained Cross-Encoder | 0.6762 | 89 ms |
| **Hybrid RRF + Fine-Tuned Cross-Encoder** | **0.7144** | **91 ms** |

---

## 6. Latency & Resource Profiling

Measurements taken on Intel Core i7 / NVIDIA RTX (CPU and GPU test beds):

| Operation | CPU (Latency P50 / P95) | GPU (Latency P50 / P95) | Memory Footprint |
|---|---:|---:|---:|
| Dense Vector Search (ChromaDB) | 28 ms / 45 ms | 12 ms / 22 ms | ~120 MB RAM |
| BM25 Retrieval | 6 ms / 12 ms | 6 ms / 12 ms | ~45 MB RAM |
| RRF Fusion (k=60 candidates) | 1 ms / 2 ms | 1 ms / 2 ms | < 5 MB RAM |
| Cross-Encoder Reranking (20 candidates) | 52 ms / 88 ms | 14 ms / 24 ms | ~180 MB VRAM / RAM |
| Evidence Stance Analysis | 350 ms / 780 ms | 350 ms / 780 ms | API-dependent |
| **End-to-End Pipeline** | **445 ms / 920 ms** | **385 ms / 840 ms** | **Total: < 350 MB** |
