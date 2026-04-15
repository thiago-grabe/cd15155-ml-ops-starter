"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient


# TODO: Test that POST /predict returns the correct response schema
# (text, sentiment, confidence, latency_ms) and a valid sentiment label.
# Hint: use @pytest.mark.parametrize to run the same assertions across multiple headlines.
def test_predict_returns_valid_response(client: TestClient, text):
    raise NotImplementedError


# TODO: Test that POST /predict/batch returns a list of results
# whose length matches the number of input texts, and each item has the correct schema.
def test_predict_batch(client: TestClient):
    raise NotImplementedError


# TODO: Test that POST /predict/batch returns HTTP 422 for an empty texts list.
def test_predict_batch_empty_list_returns_422(client: TestClient):
    raise NotImplementedError
