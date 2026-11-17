"""Centralized, environment-overridable configuration for tunable pipeline
parameters (retrieval weights, chunking sizes, model names, thresholds, ...).

This module intentionally does NOT hold secrets (API keys) — those stay as
os.getenv() calls at the call sites, loaded via python-dotenv, exactly as
before. config.py is a single source of truth for the magic numbers/strings
that used to be scattered across src/retriever.py, src/generator.py,
src/chunker.py, src/indexer.py and main.py, with the SAME defaults they had
before this module existed. Every field can be overridden via an environment
variable of the same name (uppercased), e.g. DENSE_WEIGHT=0.6.
"""

import os
from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root so relative default paths work regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class RerankerModel(str, Enum):
    """Which cross-encoder reranker to use."""
    PRETRAINED = "pretrained"
    FINETUNED = "finetuned"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Hybrid retrieval fusion (src/retriever.py: HybridRetriever) ---
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    retrieval_depth: int = 60
    rrf_k: int = 60
    top_n: int = 20
    final_k: int = 5

    # --- Generation / confidence (src/generator.py: AdvancedRAGSystem) ---
    confidence_threshold: float = 0.75

    # --- Chunking (src/chunker.py: Chunker) ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Models (main.py, src/indexer.py, src/evaluator.py) ---
    llm_provider: str = "groq"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_base: str = "https://api.groq.com/openai/v1"
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    # --- Embeddings (local HuggingFace / SentenceTransformers or OpenAI) ---
    embedding_provider: str = "local"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model: str = "text-embedding-3-small"

    # --- Reranker model selection ---
    reranker_model: RerankerModel = RerankerModel.FINETUNED
    pretrained_reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    finetuned_reranker_path: Optional[str] = None  # auto-detected if None

    # --- Temperatures ---
    generator_temperature: float = 0
    judge_temperature: float = 0
    synthetic_temperature: float = 0.7

    # --- Project paths ---
    project_root: str = _PROJECT_ROOT
    notebook_dir: str = os.path.join(_PROJECT_ROOT, "Notebook")
    data_dir: str = os.path.join(_PROJECT_ROOT, "data")
    results_dir: str = os.path.join(_PROJECT_ROOT, "results")

    def get_finetuned_model_path(self) -> Optional[str]:
        """Return the resolved path to the fine-tuned checkpoint directory.

        If finetuned_reranker_path is explicitly set, use that. Otherwise
        look in the default Notebook location.  Returns None when the
        checkpoint cannot be found (the caller should fall back to the
        pretrained model and log a warning).
        """
        if self.finetuned_reranker_path:
            path = self.finetuned_reranker_path
            if os.path.isdir(path):
                return path
            # Try relative to project root
            abs_path = os.path.join(self.project_root, path)
            if os.path.isdir(abs_path):
                return abs_path
            return None

        # Default location
        default = os.path.join(
            self.notebook_dir,
            "researchnexus-finetuning",
            "minilm-scifact-finetuned",
        )
        if os.path.isdir(default):
            return default
        return None

    def get_active_reranker_name_or_path(self) -> str:
        """Return the model name (pretrained) or local path (fine-tuned)
        that the retriever should load."""
        if self.reranker_model == RerankerModel.FINETUNED:
            ft_path = self.get_finetuned_model_path()
            if ft_path:
                return ft_path
            # Checkpoint not found — fall back to pretrained
            import logging
            logging.getLogger(__name__).warning(
                "Fine-tuned checkpoint not found; falling back to pretrained reranker."
            )
        return self.pretrained_reranker_name


settings = Settings()
