# Neural Reranker Fine-Tuning: SciFact & PyTorch Pipeline

This document details the complete experimental setup, training methodology, negative sampling strategies, and optimization dynamics for the fine-tuned neural cross-encoder integrated into **ResearchNexus**.

---

## 1. Executive Summary

ResearchNexus fine-tunes `cross-encoder/ms-marco-MiniLM-L-6-v2` specifically for scientific claim-evidence retrieval using the **SciFact** benchmark.

### Key Results
- **SciFact In-Domain (Val & Test)**: Fine-tuning yields statistically significant improvements:
  - SciFact Validation (122 queries): **+0.0417 nDCG@10** (0.6746 -> 0.7162, $p < 0.05$, 95% CI $[+0.0047, +0.0784]$)
  - SciFact Test (300 queries): **+0.0382 nDCG@10** (0.6762 -> 0.7144, $p < 0.001$, 95% CI $[+0.0135, +0.0642]$)
- **SCIDOCS Generalization**: Transfer to SCIDOCS document retrieval dropped:
  - Pretrained baseline: **0.1634 nDCG@10**
  - Fine-tuned checkpoint: **0.1457 nDCG@10** ($-0.0177$, 95% CI $[-0.0230, -0.0124]$)
- **Negative Mining Ablation**: Random negative mining unexpectedly outperformed BM25 hard negatives:
  - BM25 Hard Negatives: **0.7144 nDCG@10**
  - Random Negatives: **0.7360 nDCG@10** (+0.0216 higher)

All artifacts, training logs, loss curves, and evaluation checkpoints reside in `Notebook/researchnexus-finetuning/`.

---

## 2. Model Architecture & Checkpoint

### 2.1 Base Model Specifications
- **Base Checkpoint**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Architecture**: `BertForSequenceClassification`
  - Layers: 6 Transformer encoder blocks
  - Hidden Size: 384 dimensions
  - Attention Heads: 12
  - Vocabulary Size: 30,522 (WordPiece)
  - Total Parameters: 22.7M parameters
  - Model Binary Size: 90.8 MB (`model.safetensors`)
- **Fine-Tuned Checkpoint Location**:
  `Notebook/researchnexus-finetuning/minilm-scifact-finetuned/`
  Files: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`.

---

## 3. Dataset & Training Data Construction

### 3.1 SciFact Benchmark
- **Domain**: Biomedical and life sciences claims.
- **Labels**: Claims paired with paper abstracts annotated for `SUPPORT` and `CONTRADICT`.
- **Query Splits**:
  - Training: 809 claim queries.
  - Validation: 122 claim queries.
  - Test: 300 claim queries.
- **Corpus**: 5,183 scientific abstracts.

### 3.2 Training Pair Formulation
Each training sample consists of:
$$(q, p^+, p^-_1, p^-_2, \dots, p^-_k)$$
Where:
- $q$: Scientific claim query.
- $p^+$: Ground-truth positive abstract annotated as providing supporting or refuting evidence.
- $p^-_i$: Negative passages from the corpus that do not provide evidence for $q$.

### 3.3 Loss Function
Trained using `MultipleNegativesRankingLoss` / `BCEWithLogitsLoss`:
$$\mathcal{L} = - \sum_{i=1}^{B} \log \frac{\exp(s(q_i, p_i^+) / \tau)}{\exp(s(q_i, p_i^+) / \tau) + \sum_{j} \exp(s(q_i, p_{i,j}^-) / \tau)}$$
Where $s(q, p)$ is the unnormalized logit output of the cross-encoder and $\tau$ is the temperature scaling parameter.

---

## 4. Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Compact, low-latency, proven reranking foundation |
| Optimizer | AdamW | Standard decoupled weight decay |
| Learning Rate | $2 \times 10^{-5}$ | Small LR to prevent catastrophic forgetting of general English |
| LR Scheduler | WarmupLinear | 10% warmup steps followed by linear decay |
| Batch Size | 16 | Fitted within GPU memory constraints |
| Epochs | 4 | Monitored with early stopping based on validation nDCG@10 |
| Weight Decay | 0.01 | Regularization |
| Max Sequence Length | 256 tokens | Captures claim + scientific abstract passage |
| Mixed Precision | FP16 | Accelerated training |

---

## 5. Training Loss Dynamics

Training loss decreased steadily across the 4 epochs without evidence of severe training instability:
- **Epoch 1**: Initial loss $0.621 \to 0.418$
- **Epoch 2**: Loss $0.418 \to 0.284$
- **Epoch 3**: Loss $0.284 \to 0.197$
- **Epoch 4**: Loss $0.197 \to 0.142$

Validation nDCG@10 peaked at epoch 3 (0.7162), which was selected as the final checkpoint.

Saved loss curve plots:
- `Notebook/researchnexus-finetuning/training_curves.png`
- `Notebook/researchnexus-finetuning/weight_interpolation.png`
- `Notebook/researchnexus-finetuning/ablation_comparison.png`

---

## 6. Ablation Study: Negative Mining Strategy

A crucial experiment evaluated how negative sampling impacts reranker fine-tuning in scientific domains:

| Negative Sampling Strategy | SciFact Test nDCG@10 | Difference vs Baseline |
|---|---:|---:|
| Pretrained Baseline | 0.6762 | Ref (0.0000) |
| **BM25 Hard Negatives** | **0.7144** | **+0.0382** |
| **Random Negatives** | **0.7360** | **+0.0598** |

### Why Did Random Negatives Outperform BM25 Hard Negatives?
In scientific claim verification (SciFact):
1. **False Negative Risk**: BM25 hard negatives retrieve passages sharing dense scientific terminology (e.g., "p53 mutation in colorectal cancer"). Some of these retrieved "negatives" actually discuss identical biological mechanisms or related valid evidence not annotated in the sparse ground truth. Penalizing the model on these unannotated positive passages injects label noise.
2. **Gradient Noise**: Random negatives provide cleaner, unambiguous negative signals across the biomedical corpus, allowing the model to learn clear domain contrast before fine-grained discrimination.

ResearchNexus transparently documents this finding and provides both ablation models in the experiment suite.

---

## 7. Model Soup & Weight Interpolation

To investigate whether the pretrained general knowledge could be blended with domain-specific scientific specialization, linear weight interpolation (Model Souping) was performed:

$$\theta_{\alpha} = (1 - \alpha) \theta_{\text{pretrained}} + \alpha \theta_{\text{finetuned}}$$

Evaluating $\alpha \in [0.0, 0.1, \dots, 1.0]$:
- $\alpha = 0.0$ (Pretrained): nDCG@10 = 0.6762
- $\alpha = 0.4$: nDCG@10 = 0.6985
- $\alpha = 0.7$: nDCG@10 = 0.7158
- $\alpha = 1.0$ (Fine-tuned): nDCG@10 = 0.7144

Weight interpolation near $\alpha \approx 0.7$ achieved the highest stability across both SciFact and SCIDOCS queries.
