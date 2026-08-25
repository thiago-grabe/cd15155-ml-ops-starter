"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# `python app/main.py` puts app/ on sys.path (flat import works, `app` package does not).
# `uvicorn app.main:app` and `pytest` put the project root on sys.path (package import works).
# Catch ModuleNotFoundError specifically: a broad ImportError would mask a genuinely
# broken app.utils (e.g. transformers missing) behind a misleading fallback.
try:
    from app.utils import load_classifier
except ModuleNotFoundError:
    from utils import load_classifier

# `format="%(message)s"` is required: the default formatter prefixes "INFO:root:",
# which would make every line invalid JSON and break log ingestion.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    """Emit one structured log record as a single-line JSON object."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level.upper(),
        "message": message,
        **kwargs,
    }
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(record))


def _get_or_create(metric_cls, name: str, documentation: str, **kwargs):
    """Build a Prometheus collector, tolerating a repeated module import.

    Collectors are process-global. tests/conftest.py places BOTH the project root and
    app/ on sys.path, so this module can be imported under two distinct names; the
    second import re-runs this body and plain construction would raise
    "Duplicated timeseries in CollectorRegistry".
    """
    try:
        return metric_cls(name, documentation, **kwargs)
    except ValueError:
        existing = getattr(REGISTRY, "_names_to_collectors", {})
        for candidate in (name, f"{name}_total"):
            if candidate in existing:
                return existing[candidate]
        raise


PREDICTION_REQUESTS = _get_or_create(
    Counter,
    "prediction_requests_total",
    "Total number of prediction requests, labelled by predicted sentiment.",
    labelnames=["sentiment"],
)
PREDICTION_LATENCY = _get_or_create(
    Histogram,
    "prediction_latency_ms",
    "Prediction latency in milliseconds.",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)
PREDICTION_ERRORS = _get_or_create(
    Counter,
    "prediction_errors_total",
    "Total number of prediction errors.",
)

load_dotenv()

classifiers = {}

MAX_TEXT_CHARS = 2000
MODEL_MAX_TOKENS = 512


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...", model_source=os.getenv("MODEL_SOURCE", "mlflow"))
    try:
        classifiers["sentiment"] = load_classifier()
        log("INFO", "Model loaded successfully")
    except Exception as e:  # noqa: BLE001 - startup must survive ANY load failure
        # Do not abort startup: an unloaded model must surface as HTTP 503 from
        # /health so the orchestrator can restart us, rather than a crash loop
        # that never binds a port and reports nothing useful.
        log("ERROR", "Model failed to load", error=str(e), error_type=type(e).__name__)
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    """Run batch inference and record Prometheus metrics for every prediction."""
    try:
        for text in texts:
            if len(text) > MAX_TEXT_CHARS:
                log(
                    "WARNING",
                    "Input exceeds 2000 characters; the model will truncate it",
                    text_length=len(text),
                    max_chars=MAX_TEXT_CHARS,
                )

        start = time.perf_counter()
        # truncation/max_length must be passed at call time: a pipeline restored from
        # the MLflow registry does not carry the kwargs evaluate.py built it with.
        raw = classifiers["sentiment"](
            texts, truncation=True, max_length=MODEL_MAX_TOKENS
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        # Per-item latency, so the histogram means "latency per prediction" identically
        # for /predict and /predict/batch.
        latency_ms = round(total_ms / max(len(texts), 1), 2)

        results = []
        for text, item in zip(texts, raw, strict=True):
            sentiment = str(item["label"]).lower()
            confidence = float(item["score"])

            PREDICTION_REQUESTS.labels(sentiment=sentiment).inc()
            # Explicit .observe() — never PREDICTION_LATENCY.time(), which records
            # SECONDS and would be 1000x wrong on a _ms metric.
            PREDICTION_LATENCY.observe(latency_ms)

            log(
                "INFO",
                "prediction",
                sentiment=sentiment,
                confidence=round(confidence, 4),
                latency_ms=latency_ms,
            )
            results.append(
                PredictionResult(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence,
                    latency_ms=latency_ms,
                )
            )
        return results
    except Exception as e:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(e), error_type=type(e).__name__)
        raise


def run_prediction(text: str) -> tuple[list[PredictionResult], float]:
    """Single-text helper: returns (predictions, measured latency in ms)."""
    predictions = run_predictions([text])
    return predictions, predictions[0].latency_ms


def _require_model() -> None:
    """Guard executed before any request validation.

    Availability is a precondition for validation: answering 422 to a request that
    could not have been served regardless of its body would be wrong, so this runs first.
    """
    if classifiers.get("sentiment") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    _require_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    _require_model()
    # Pydantic accepts "" for a str field, so the empty case never reaches FastAPI's
    # automatic 422 machinery and must be raised by hand. .strip() also rejects
    # whitespace-only input, which would otherwise yield a meaningless prediction.
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    predictions, _latency_ms = run_prediction(request.text)
    return predictions[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    _require_model()
    # Likewise, list[str] accepts [] — Pydantic will not reject it for us.
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")
    if any(not t.strip() for t in request.texts):
        raise HTTPException(
            status_code=422, detail="texts must not contain empty strings"
        )
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
