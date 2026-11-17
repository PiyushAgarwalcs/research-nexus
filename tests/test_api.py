import pytest

from app import (
    ModelSelectRequest,
    QueryRequest,
    get_evaluation_results,
    get_evaluation_summary,
    health,
    list_documents,
    list_models,
    metrics,
    query,
    select_model,
)
from fastapi import HTTPException


def test_health_endpoint():
    data = health()
    assert data["status"] == "ok"
    assert data["service"] == "ResearchNexus"
    assert "finetuned_checkpoint_available" in data
    assert data["finetuned_checkpoint_available"] is True


def test_metrics_endpoint():
    data = metrics()
    assert "requests_served" in data


def test_models_endpoint():
    data = list_models()
    assert "active_model" in data
    assert "available_models" in data
    model_ids = [m["id"] for m in data["available_models"]]
    assert "finetuned" in model_ids
    assert "pretrained" in model_ids


def test_evaluation_summary_endpoint():
    data = get_evaluation_summary()
    assert "key_conclusions" in data
    assert len(data["key_conclusions"]) >= 4
    topics = [c["topic"] for c in data["key_conclusions"]]
    assert any("In-Domain" in t for t in topics)
    assert any("Generalization" in t for t in topics)


def test_evaluation_results_endpoint():
    data = get_evaluation_results()
    assert "final_comparison" in data
    assert "training_history" in data
    assert len(data["training_history"]) > 0


def test_documents_list_endpoint():
    docs = list_documents()
    assert isinstance(docs, list)


def test_query_validation_empty_query():
    with pytest.raises(HTTPException) as excinfo:
        query(QueryRequest(query=""))
    assert excinfo.value.status_code in (400, 503)


def test_select_model_uninitialized_pipeline():
    with pytest.raises(HTTPException) as excinfo:
        select_model(ModelSelectRequest(model="pretrained"))
    assert excinfo.value.status_code == 503
