"""FastAPI service exposing ResearchNexus — Scientific Research & Evidence Intelligence Platform.

Endpoints:
    POST   /query             -> Research QA, citations, confidence, evidence analysis
    GET    /health            -> Service health & active model info
    GET    /metrics           -> Rolling latency & cost metrics
    GET    /models            -> List available reranker models & active selection
    POST   /models/select     -> Switch active reranker (pretrained vs fine-tuned)
    GET    /documents         -> List indexed research documents & metadata
    POST   /documents/upload  -> Upload new research paper (PDF, TXT, MD, HTML)
    DELETE /documents/{name}  -> Delete a document from corpus
    POST   /documents/index   -> Trigger index rebuilding
    GET    /evaluation/results-> Experiment metrics & benchmark comparison
    GET    /evaluation/summary-> Executive scientific summary & CI findings
"""

import logging
import os
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from src.loader import MultiLoader
from src.pipeline import build_pipeline
from src.results_manager import results_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("researchnexus.app")

_PRICE_PER_INPUT_TOKEN = 0.15 / 1_000_000
_PRICE_PER_OUTPUT_TOKEN = 0.60 / 1_000_000
_MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB

_TOKEN_ENCODING = None


def _get_encoding():
    global _TOKEN_ENCODING
    if _TOKEN_ENCODING is None:
        try:
            _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKEN_ENCODING = False
    return _TOKEN_ENCODING if _TOKEN_ENCODING is not False else None


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoding()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


_metrics: Dict[str, deque] = {
    "latency_ms": deque(maxlen=1000),
    "cost_usd": deque(maxlen=1000),
}
_state: Dict[str, Any] = {"pipeline": None}


def _estimate_cost_usd(prompt_text: str, completion_text: str) -> float:
    input_tokens = _count_tokens(prompt_text)
    output_tokens = _count_tokens(completion_text)
    return (
        input_tokens * _PRICE_PER_INPUT_TOKEN
        + output_tokens * _PRICE_PER_OUTPUT_TOKEN
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    logger.info("Initializing ResearchNexus API (data_dir=%r)...", data_dir)
    if api_key:
        try:
            start = time.perf_counter()
            _state["pipeline"] = build_pipeline(
                openai_api_key=api_key,
                data_dir=data_dir,
                reranker_model=settings.reranker_model,
            )
            elapsed = time.perf_counter() - start
            logger.info("ResearchNexus pipeline ready in %.2fs.", elapsed)
        except Exception as e:
            logger.warning("Pipeline deferred initialization: %s", e)
    else:
        logger.warning("Neither GROQ_API_KEY nor OPENAI_API_KEY set. Pipeline will initialize upon key configuration.")

    yield

    _state["pipeline"] = None


app = FastAPI(
    title="ResearchNexus API",
    description="Scientific Research & Evidence Intelligence Platform API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Schemas ---

class QueryRequest(BaseModel):
    query: str = Field(description="Scientific question or research inquiry")
    reranker_model: Optional[str] = Field(
        default=None,
        description="Optional model override: 'pretrained' or 'finetuned'",
    )


class QueryResponse(BaseModel):
    answer: str
    status: str
    latency_ms: float
    confidence: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    verified_citations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_analysis: Optional[Dict[str, Any]] = None
    active_model: Dict[str, str] = Field(default_factory=dict)


class ModelSelectRequest(BaseModel):
    model: str = Field(description="'pretrained' or 'finetuned' (or local checkpoint path)")


class DocumentInfo(BaseModel):
    filename: str
    file_size_bytes: int
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    doc_id: Optional[str] = None


# --- Endpoints ---

@app.get("/health")
def health() -> Dict[str, Any]:
    pipeline = _state.get("pipeline")
    retriever = getattr(pipeline, "retriever", None) if pipeline else None
    return {
        "status": "ok",
        "service": "ResearchNexus",
        "pipeline_ready": pipeline is not None,
        "active_reranker": getattr(retriever, "model_label", "Not loaded"),
        "finetuned_checkpoint_available": settings.get_finetuned_model_path() is not None,
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    latencies = _metrics["latency_ms"]
    costs = _metrics["cost_usd"]
    return {
        "requests_served": len(latencies),
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
    }


@app.get("/models")
def list_models() -> Dict[str, Any]:
    pipeline = _state.get("pipeline")
    retriever = getattr(pipeline, "retriever", None) if pipeline else None
    ft_path = settings.get_finetuned_model_path()

    return {
        "active_model": {
            "target": getattr(retriever, "model_target", settings.reranker_model.value),
            "label": getattr(retriever, "model_label", "Default Reranker"),
        },
        "available_models": [
            {
                "id": "finetuned",
                "label": "Fine-tuned MiniLM (SciFact)",
                "type": "CrossEncoder",
                "path": ft_path,
                "is_local": True,
                "available": ft_path is not None and os.path.isdir(ft_path),
                "description": "MiniLM reranker fine-tuned on scientific claim-evidence pairs (SciFact)",
            },
            {
                "id": "pretrained",
                "label": "Pretrained MiniLM (MS-MARCO)",
                "type": "CrossEncoder",
                "name": settings.pretrained_reranker_name,
                "is_local": False,
                "available": True,
                "description": "Standard pretrained cross-encoder/ms-marco-MiniLM-L-6-v2 baseline",
            },
        ],
    }


@app.post("/models/select")
def select_model(request: ModelSelectRequest) -> Dict[str, Any]:
    pipeline = _state.get("pipeline")
    if not pipeline or not hasattr(pipeline, "retriever"):
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        new_label = pipeline.retriever.switch_reranker(request.model)
        return {
            "status": "success",
            "message": f"Active reranker switched to: {new_label}",
            "active_model": {
                "target": pipeline.retriever.model_target,
                "label": pipeline.retriever.model_label,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to switch reranker: {e}")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. Ensure OPENAI_API_KEY is configured and documents are indexed.",
        )

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must be a non-empty string.")

    # NOTE: In a multi-worker deployment, switching the retriever model on a shared
    # singleton pipeline is not thread-safe. A per-request retriever or thread lock
    # should be used if serving concurrent requests with different model overrides.
    if request.reranker_model and hasattr(pipeline, "retriever"):
        try:
            pipeline.retriever.switch_reranker(request.reranker_model)
        except Exception as e:
            logger.warning("Reranker switch requested in query failed: %s", e)

    start = time.perf_counter()
    result = pipeline.generate_robust_answer(request.query)
    latency_ms = (time.perf_counter() - start) * 1000

    status = result.get("status", "Unknown")

    if status == "Success":
        answer = result.get("answer", "")
        flagged_citations = result.get("flagged_citations", [])
        verified_citations = result.get("verified_citations", [])
        confidence = result.get("confidence_metrics")
        evidence_analysis = result.get("evidence_analysis")
        active_model = result.get("active_model", {})
    else:
        answer = result.get("found_context", "")
        flagged_citations = []
        verified_citations = []
        confidence = None
        evidence_analysis = None
        active_model = result.get("active_model", {})

    retrieved_chunks = result.get("retrieved_chunks", [])
    context_text = "\n".join(c.page_content for c in retrieved_chunks) if retrieved_chunks else ""
    cost_usd = _estimate_cost_usd(context_text + request.query, answer)

    _metrics["latency_ms"].append(latency_ms)
    _metrics["cost_usd"].append(cost_usd)

    return QueryResponse(
        answer=answer,
        status=status,
        latency_ms=round(latency_ms, 2),
        confidence=confidence,
        citations=flagged_citations,
        verified_citations=verified_citations,
        evidence_analysis=evidence_analysis,
        active_model=active_model,
    )


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents() -> List[DocumentInfo]:
    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    if not os.path.exists(data_dir):
        return []

    loader = MultiLoader(data_dir)
    documents = loader.load_documents(deduplicate=False)
    results = []
    for doc in documents:
        m = doc.metadata
        results.append(
            DocumentInfo(
                filename=m.get("filename", "unknown"),
                file_size_bytes=m.get("file_size", 0),
                title=m.get("title"),
                authors=m.get("authors"),
                year=m.get("year"),
                doi=m.get("doi"),
                abstract=m.get("abstract"),
                doc_id=m.get("doc_id"),
            )
        )
    return results


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md", ".html", ".htm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: .pdf, .txt, .md, .html",
        )

    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    destination = os.path.join(data_dir, file.filename)

    # Stream write with size limit
    size = 0
    with open(destination, "wb") as buffer:
        while chunk := await file.read(1024 * 64):
            size += len(chunk)
            if size > _MAX_UPLOAD_SIZE:
                buffer.close()
                os.remove(destination)
                raise HTTPException(status_code=413, detail=f"File exceeds {_MAX_UPLOAD_SIZE // (1024*1024)}MB limit.")
            buffer.write(chunk)

    # Validate and inspect scientific metadata
    loader = MultiLoader(data_dir)
    doc = loader.load_file(destination)
    if not doc:
        if os.path.exists(destination):
            os.remove(destination)
        raise HTTPException(status_code=400, detail="File could not be parsed or contains no readable text.")

    return {
        "status": "success",
        "message": f"Uploaded and parsed {file.filename} successfully.",
        "metadata": doc.metadata,
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str) -> Dict[str, str]:
    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    target = os.path.join(data_dir, filename)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    try:
        os.remove(target)
        return {"status": "success", "message": f"Deleted '{filename}' successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")


@app.post("/documents/index")
def trigger_index() -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Neither GROQ_API_KEY nor OPENAI_API_KEY is configured.",
        )

    data_dir = os.getenv("DATA_DIR", settings.data_dir)
    try:
        start = time.perf_counter()
        _state["pipeline"] = build_pipeline(
            openai_api_key=api_key,
            data_dir=data_dir,
            reranker_model=settings.reranker_model,
        )
        elapsed = time.perf_counter() - start
        return {
            "status": "success",
            "message": f"Rebuilt hybrid index in {elapsed:.2f}s.",
            "data_dir": data_dir,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.get("/evaluation/results")
def get_evaluation_results() -> Dict[str, Any]:
    raw_results = results_manager.get_evaluation_results()
    comparison_df = results_manager.get_model_comparison_table()
    stats_df = results_manager.get_statistical_significance()
    neg_ablation_df = results_manager.get_negative_ablation()
    weight_interp_df = results_manager.get_weight_interpolation()
    training_history = results_manager.get_training_history()

    return {
        "evaluation_metrics": raw_results,
        "final_comparison": comparison_df.to_dict(orient="records") if not comparison_df.empty else [],
        "statistical_significance": stats_df.to_dict(orient="records") if not stats_df.empty else [],
        "negative_ablation": neg_ablation_df.to_dict(orient="records") if not neg_ablation_df.empty else [],
        "weight_interpolation": weight_interp_df.to_dict(orient="records") if not weight_interp_df.empty else [],
        "training_history": training_history,
    }


@app.get("/evaluation/summary")
def get_evaluation_summary() -> Dict[str, Any]:
    return results_manager.get_executive_summary()
