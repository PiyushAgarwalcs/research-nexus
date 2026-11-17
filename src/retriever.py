import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from sentence_transformers import CrossEncoder

from config import RerankerModel, settings

logger = logging.getLogger(__name__)

# Pretrained model name preserved for backward compatibility
_RERANKER_MODEL = settings.pretrained_reranker_name

# In-memory cache of loaded models to avoid duplicate downloads and disk I/O
_LOADED_MODELS: Dict[str, CrossEncoder] = {}


def get_default_device() -> str:
    """Return 'cuda' if GPU is available and initialized, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_reranker_target(model_choice: Optional[Union[str, RerankerModel]] = None) -> Tuple[str, str]:
    """Resolve a model choice or enum to (resolved_path_or_name, human_label).

    Supports:
      - None -> active model configured in settings
      - RerankerModel.PRETRAINED / 'pretrained' -> HuggingFace pretrained model
      - RerankerModel.FINETUNED / 'finetuned' -> Local fine-tuned checkpoint
      - Any explicit local directory path or HuggingFace repo ID
    """
    if model_choice is None:
        model_choice = settings.reranker_model

    if isinstance(model_choice, RerankerModel):
        model_choice = model_choice.value

    choice_lower = str(model_choice).lower().strip()

    if choice_lower in ("pretrained", "ms-marco", "base"):
        return settings.pretrained_reranker_name, "Pretrained MiniLM (ms-marco-MiniLM-L-6-v2)"

    if choice_lower in ("finetuned", "fine-tuned", "scifact"):
        ft_path = settings.get_finetuned_model_path()
        if ft_path and os.path.isdir(ft_path):
            return ft_path, "Fine-tuned MiniLM (SciFact checkpoint)"
        logger.warning(
            "Fine-tuned checkpoint not found at %s. Falling back to pretrained model.",
            ft_path,
        )
        return settings.pretrained_reranker_name, "Pretrained MiniLM (ms-marco-MiniLM-L-6-v2) [Fallback]"

    # If it is a direct path to an existing directory
    if os.path.isdir(str(model_choice)):
        return str(model_choice), f"Local Model ({os.path.basename(str(model_choice))})"

    # Otherwise treat as HuggingFace identifier or custom path
    return str(model_choice), str(model_choice)


def _load_cross_encoder(
    model_name_or_path: Optional[str] = None,
    max_retries: int = 5,
    device: Optional[str] = None,
) -> CrossEncoder:
    """Load a CrossEncoder with caching, local checkpoint support, and network retries.

    If model_name_or_path is None, loads the configured active model.
    Loaded models are cached by (path, device) to prevent duplicate weight loading.
    """
    if model_name_or_path is None:
        target, _ = resolve_reranker_target()
    else:
        target, _ = resolve_reranker_target(model_name_or_path)

    dev = device or get_default_device()
    cache_key = f"{target}::{dev}"

    if cache_key in _LOADED_MODELS:
        logger.debug("Reusing cached CrossEncoder for %s on %s", target, dev)
        return _LOADED_MODELS[cache_key]

    is_local = os.path.isdir(target)

    if is_local:
        logger.info("Loading local fine-tuned CrossEncoder from %s on %s...", target, dev)
        try:
            model = CrossEncoder(target, device=dev)
            _LOADED_MODELS[cache_key] = model
            return model
        except Exception as e:
            logger.error("Failed to load local model from %s: %s", target, e)
            if target != settings.pretrained_reranker_name:
                logger.info("Falling back to pretrained reranker: %s", settings.pretrained_reranker_name)
                return _load_cross_encoder(settings.pretrained_reranker_name, max_retries=max_retries, device=dev)
            raise

    # Remote model with download retry logic
    logger.info("Loading remote CrossEncoder %s on %s...", target, dev)
    for attempt in range(1, max_retries + 1):
        try:
            model = CrossEncoder(target, device=dev)
            _LOADED_MODELS[cache_key] = model
            return model
        except Exception as e:
            if attempt < max_retries:
                wait = 5 * attempt
                logger.warning(
                    "CrossEncoder download failed (attempt %d/%d), retrying in %ds: %s",
                    attempt,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                logger.error("Failed to load CrossEncoder after %d attempts: %s", max_retries, e)
                raise
    raise RuntimeError("Unreachable")


class HybridRetriever:
    def __init__(
        self,
        vectorstore,
        bm25_retriever,
        dense_weight: float = settings.dense_weight,
        sparse_weight: float = settings.sparse_weight,
        use_reranker: bool = True,
        reranker=None,
        reranker_model: Optional[Union[str, RerankerModel]] = None,
        device: Optional[str] = None,
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranker = use_reranker
        self.device = device or get_default_device()

        # Determine target model identity
        if reranker is not None:
            self.reranker = reranker if use_reranker else None
            self.model_target = "custom_injected"
            self.model_label = "Custom Injected Reranker"
        elif use_reranker:
            target, label = resolve_reranker_target(reranker_model)
            self.model_target = target
            self.model_label = label
            self.reranker = _load_cross_encoder(target, device=self.device)
        else:
            self.reranker = None
            self.model_target = "disabled"
            self.model_label = "Reranker Disabled (RRF only)"

    def switch_reranker(self, model_choice: Union[str, RerankerModel]) -> str:
        """Dynamically switch between pretrained and fine-tuned models at runtime.

        Returns the human-readable label of the newly active model.
        """
        target, label = resolve_reranker_target(model_choice)
        self.model_target = target
        self.model_label = label
        self.use_reranker = True
        self.reranker = _load_cross_encoder(target, device=self.device)
        logger.info("Retriever reranker switched to: %s (%s)", label, target)
        return label

    def retrieve_and_fuse(
        self,
        query,
        retrieval_depth=settings.retrieval_depth,
        rrf_k=settings.rrf_k,
        top_n=settings.top_n,
    ):
        """
        retrieval_depth: how many candidates to pull from each of the dense/
        sparse retrievers before fusion. rrf_k: the smoothing constant in the
        Reciprocal Rank Fusion formula 1/(rrf_k + rank) — independent of how
        deep the initial retrieval goes. Ranks are 1-based, as in the standard
        RRF formulation (so rrf_k=0 is valid and the top hit scores 1/(rrf_k+1)).
        """
        dense_results = self.vectorstore.similarity_search(query, k=retrieval_depth)
        sparse_results = self.bm25_retriever.invoke(query)[:retrieval_depth]

        fused_scores = {}

        for results, weight in (
            (dense_results, self.dense_weight),
            (sparse_results, self.sparse_weight),
        ):
            # A chunk contributes once per retriever, at its best rank — a
            # duplicate in one result list must not add its score twice.
            seen = set()
            for rank, doc in enumerate(results, start=1):
                doc_content = doc.page_content
                if doc_content in seen:
                    continue
                seen.add(doc_content)
                if doc_content not in fused_scores:
                    fused_scores[doc_content] = {"doc": doc, "score": 0}
                fused_scores[doc_content]["score"] += weight * (1 / (rrf_k + rank))

        ranked_results = sorted(
            fused_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [item for item in ranked_results[:top_n]]

    def rerank(self, query, candidates, final_k=settings.final_k):
        """Returns [(doc, cross_encoder_score), ...] sorted by score, so callers
        can use the score as a real relevance signal instead of discarding it."""
        if not candidates:
            return []
        pairs =[[query, doc["doc"].page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [(item["doc"], float(score)) for item, score in scored_docs[:final_k]]

    def get_relevant_documents(self, query):
        candidates = self.retrieve_and_fuse(query, top_n=settings.top_n)
        if not self.use_reranker:
            # Return the top final_k straight from RRF fusion, using the
            # fused score in place of the cross-encoder score so the
            # (doc, score) contract stays intact for downstream callers
            # (e.g. score_confidence's rerank_scores parameter). Note these
            # are not calibrated relevance logits: RRF scores are ~0.01, so
            # score_confidence's sigmoid maps them all to ~0.5.
            top = candidates[: settings.final_k]
            return [(item["doc"], float(item["score"])) for item in top]
        return self.rerank(query, candidates, final_k=settings.final_k)
