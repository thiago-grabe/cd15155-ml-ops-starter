"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_classifier

# TODO (Task 6)
# Configure the Python logger using basicConfig.


# TODO (Task 6)
# Implement a structured logging helper that emits each log entry as a JSON
# object containing at least a timestamp, level, and message field.
def log(level: str, message: str, **kwargs) -> None:
    pass


# TODO (Task 6)
# Define the following Prometheus metrics:
# 1. A Counter for the total number of prediction requests, labelled by sentiment.
# 2. A Histogram for prediction latency in milliseconds.
# 3. A Counter for the total number of prediction errors.
# Then instrument run_predictions() to update each metric accordingly.
# Documentation: https://prometheus.io/docs/concepts/metric_types/
PREDICTION_REQUESTS = None
PREDICTION_LATENCY = None
PREDICTION_ERRORS = None

load_dotenv()

classifiers = {}


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
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        # TODO (Task 3): Log a warning using log() if any input text exceeds 2000
        # characters — the model will silently truncate it, so this makes it visible.

        # TODO (Task 3): Record the start time, run batch inference using
        # classifiers["sentiment"], and compute latency_ms from start to finish.

        # TODO (Task 3): Build and return a list of PredictionResult objects from
        # the inference results (each result has "label" and "score" keys).

        # TODO (Task 6): Observe latency_ms on PREDICTION_LATENCY and increment
        # PREDICTION_REQUESTS (labelled by sentiment) for each prediction.

        # TODO (Task 6): Log each prediction using log() with sentiment, confidence,
        # and latency_ms fields.
        raise NotImplementedError
    except Exception as e:
        # TODO (Task 6): Increment PREDICTION_ERRORS, log the error using log(),
        # then re-raise.
        raise e


# TODO (Task 6): Implement the /metrics endpoint.
# Return the Prometheus metrics in the correct format using generate_latest()
# and CONTENT_TYPE_LATEST.
# Documentation: https://prometheus.io/docs/instrumenting/exposition_formats/
@app.get("/metrics")
def metrics():
    raise NotImplementedError


# TODO: Implement the /health endpoint.
# Return {"status": "ok"} when the model
# is loaded, and raise an HTTP 503 error when it is not.
@app.get("/health")
def health():
    raise NotImplementedError


# TODO: Implement the POST /predict endpoint.
# Accept a PredictRequest and return a PredictionResult.
# Return HTTP 503 if the model is not loaded
# Return HTTP 422 if the text is empty
@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    raise NotImplementedError


# TODO: Implement the POST /predict/batch endpoint.
# Accept a PredictBatchRequest and return a list of PredictionResult.
# Return HTTP 503 if the model is not loaded
# and HTTP 422 if the texts list is empty.
@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    raise NotImplementedError


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
