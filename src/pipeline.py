"""Factory that wires together the full Hybrid RAG pipeline: load documents
-> chunk -> index (dense + BM25) -> HybridRetriever -> AdvancedRAGSystem.

This mirrors the setup that main.py performs for its single-query demo, so
app.py, streamlit_app.py, and any future caller can build the same pipeline
without duplicating that wiring. Uses config.py for every tunable default;
callers can still override anything they need to.
"""

import logging
import os
from typing import Optional, Union

from config import RerankerModel, settings
from src.chunker import Chunker
from src.generator import AdvancedRAGSystem
from src.indexer import Indexer
from src.loader import MultiLoader
from src.models import get_llm
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def build_pipeline(
    openai_api_key: Optional[str] = None,
    data_dir: str = "./data/",
    persist_directory: str = "./chroma_main_db",
    chunking_strategy: str = "TokenRecursive",
    max_documents: int = 50,
    reranker_model: Optional[Union[str, RerankerModel]] = None,
    dense_weight: float = settings.dense_weight,
    sparse_weight: float = settings.sparse_weight,
    use_reranker: bool = True,
) -> AdvancedRAGSystem:
    """Build a ready-to-query AdvancedRAGSystem with fine-tuned / pretrained model choice.

    Loads documents from `data_dir`, chunks them, builds a Chroma + BM25
    index at `persist_directory`, wraps it in a HybridRetriever (configured
    with the selected reranker), and returns an AdvancedRAGSystem ready to
    call `generate_robust_answer(query)` on.
    """
    key = openai_api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Please configure GROQ_API_KEY (for 100% free Groq) "
            "or OPENAI_API_KEY in your .env file."
        )

    loader = MultiLoader(data_dir)
    all_documents = loader.load_documents(deduplicate=True)
    if not all_documents:
        raise ValueError(
            f"No documents loaded from {data_dir!r}. Run `python scripts/fetch_corpus.py` to download the default corpus or upload papers."
        )
    documents = all_documents[:max_documents]

    logger.info("Loaded %d document(s) from %s", len(documents), data_dir)

    chunker = Chunker(embedding_fn=None)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(
            chunker.chunk_documents(
                doc.page_content, doc.metadata, strategy=chunking_strategy
            )
        )
    logger.info("Produced %d chunk(s) using the %r strategy", len(all_chunks), chunking_strategy)

    idx = Indexer(api_key=key)
    vectorstore, bm25_retriever = idx.create_indexes(
        all_chunks, persist_directory=persist_directory
    )

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )

    generator_llm = get_llm(
        temperature=settings.generator_temperature,
        api_key=key,
    )

    return AdvancedRAGSystem(llm=generator_llm, retriever=retriever)
