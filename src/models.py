"""Provider abstraction for 100% free offline/open-source execution or cloud APIs.

Supports:
- LLM: Groq (ultra-fast free Llama-3.3-70b inference via OpenAI-compatible endpoint) or OpenAI.
- Embeddings: Local SentenceTransformers (all-MiniLM-L6-v2, 100% free, CPU/GPU) or OpenAI.
"""

import logging
import os
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Free, offline local sentence embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local SentenceTransformer embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        res = self.embed_documents([text])
        return res[0] if res else []


def get_embeddings(model_name: Optional[str] = None) -> Embeddings:
    """Return embedding instance based on configuration and environment.

    Defaults to LocalSentenceTransformerEmbeddings (100% free, no OpenAI API
    needed) whenever embedding_provider is 'local' or OPENAI_API_KEY is not set.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    provider = getattr(settings, "embedding_provider", "local").lower()

    if provider == "local" or not openai_key:
        chosen = model_name or getattr(settings, "local_embedding_model", "all-MiniLM-L6-v2")
        return LocalSentenceTransformerEmbeddings(model_name=chosen)

    from langchain_openai import OpenAIEmbeddings
    chosen = model_name or settings.embedding_model
    return OpenAIEmbeddings(model=chosen, api_key=openai_key)


def get_llm(
    temperature: float = 0.0,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """Return configured Chat LLM instance.

    Defaults to Groq (100% free tier Llama-3.3-70b) if GROQ_API_KEY is set
    or llm_provider is 'groq'. Falls back to OpenAI if OPENAI_API_KEY is set.
    """
    groq_key = api_key or os.getenv("GROQ_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    provider = getattr(settings, "llm_provider", "groq").lower()

    if groq_key and (provider == "groq" or not openai_key):
        chosen_model = model or getattr(settings, "groq_model", "llama-3.3-70b-versatile")
        base_url = getattr(settings, "groq_api_base", "https://api.groq.com/openai/v1")
        logger.info("Initialized Groq LLM: %s", chosen_model)
        return ChatOpenAI(
            model=chosen_model,
            api_key=groq_key,
            base_url=base_url,
            temperature=temperature,
        )

    if openai_key:
        chosen_model = model or settings.generator_model
        logger.info("Initialized OpenAI LLM: %s", chosen_model)
        return ChatOpenAI(
            model=chosen_model,
            api_key=openai_key,
            temperature=temperature,
        )

    raise ValueError(
        "No LLM API key configured. Please set GROQ_API_KEY in your .env file "
        "to run with Groq (100% free), or set OPENAI_API_KEY."
    )
