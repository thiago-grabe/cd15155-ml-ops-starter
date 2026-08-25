"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# `python app/main.py` puts app/ on sys.path (flat import works, `app` package does not).
# `uvicorn app.main:app` and `pytest` put the project root on sys.path (package import works).
# Catch ModuleNotFoundError specifically: a broad ImportError would mask a genuinely
# broken app.utils (e.g. transformers missing) behind a misleading fallback.
try:
    from app.utils import load_classifier
except ModuleNotFoundError:
    from utils import load_classifier

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    """Emit a log record. Upgraded to structured JSON in the monitoring change."""
    extra = "  ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.log(getattr(logging, level.upper(), logging.INFO), f"{message} {extra}".strip())


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
    except Exception as e:
        # Do not abort startup: an unloaded model must surface as HTTP 503 from
        # /health so the orchestrator can restart us, rather than a crash loop
        # that never binds a port and reports nothing useful.
        log("ERROR", "Model failed to load", error=str(e), error_type=type(e).__name__)
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    """Run batch inference and return one result per input text."""
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
        # Per-item latency, so the number means "latency per prediction" identically
        # for /predict and /predict/batch.
        latency_ms = round(total_ms / max(len(texts), 1), 2)

        results = []
        for text, item in zip(texts, raw):
            sentiment = str(item["label"]).lower()
            confidence = float(item["score"])
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
    return run_predictions([request.text])[0]


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
        port=int(os.getenv("API_PORT", 8000)),
    )
