"""
Integration tests for the Sentiment Analysis API.

The `client` fixture (tests/conftest.py) is session-scoped, so FinBERT is loaded
exactly once and held for the whole run.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import classifiers

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
RESULT_FIELDS = {"text", "sentiment", "confidence", "latency_ms"}

HEADLINES = [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
    "Stocks closed flat on Friday amid low trading volume.",
]


def _assert_valid_result(payload: dict, expected_text: str) -> None:
    assert set(payload) == RESULT_FIELDS
    assert payload["text"] == expected_text
    assert payload["sentiment"] in VALID_SENTIMENTS
    assert isinstance(payload["confidence"], float)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["latency_ms"] > 0


@pytest.mark.parametrize("text", HEADLINES)
def test_predict_returns_valid_response(client: TestClient, text):
    """POST /predict returns the full result schema and a valid sentiment label."""
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    _assert_valid_result(response.json(), text)


def test_predict_batch(client: TestClient):
    """POST /predict/batch returns one correctly-shaped result per input text."""
    response = client.post("/predict/batch", json={"texts": HEADLINES})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    assert len(body) == len(HEADLINES)
    for item, expected_text in zip(body, HEADLINES):
        _assert_valid_result(item, expected_text)


def test_predict_batch_empty_list_returns_422(client: TestClient):
    """An empty texts list is rejected with HTTP 422."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_empty_text_returns_422(client: TestClient):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_whitespace_only_text_returns_422(client: TestClient):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_endpoints_return_503_when_model_unloaded(client: TestClient):
    """With the model unloaded, every serving endpoint reports 503 — including for
    payloads that would otherwise be rejected as 422."""
    saved = classifiers.pop("sentiment", None)
    assert saved is not None, "model fixture was not loaded"
    try:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"text": HEADLINES[0]}).status_code == 503
        assert (
            client.post("/predict/batch", json={"texts": HEADLINES}).status_code == 503
        )
        # Availability is checked before validation, so an empty body is still 503.
        assert client.post("/predict", json={"text": ""}).status_code == 503
    finally:
        classifiers["sentiment"] = saved


def test_metrics_endpoint_exposes_prometheus_metrics(client: TestClient):
    # Ensure at least one prediction has been recorded, so the labelled counter exists.
    client.post("/predict", json={"text": HEADLINES[0]})

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    assert "prediction_requests_total" in body
    assert "prediction_latency_ms" in body
    assert "prediction_errors_total" in body
