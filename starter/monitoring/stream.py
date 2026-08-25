"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# API_HOST is a BIND address (0.0.0.0) and is meaningless as a request target, so an
# explicit API_URL takes precedence. Inside compose this is http://api:8000.
API_URL = os.getenv(
    "API_URL",
    f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100    # delay between requests to simulate real traffic (ms)


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    n = len(window)
    if n == 0:
        return

    counts = Counter(r["sentiment"] for r in window)
    metrics = {
        "pct_positive": 100.0 * counts.get("positive", 0) / n,
        "pct_negative": 100.0 * counts.get("negative", 0) / n,
        "pct_neutral": 100.0 * counts.get("neutral", 0) / n,
        "avg_confidence": sum(r["confidence"] for r in window) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in window) / n,
        "window_size": n,
    }

    # step= is what makes these a time series in the MLflow UI. Without it every
    # window would overwrite the last and you'd see a single scalar.
    mlflow.log_metrics(metrics, step=window_idx)

    summary = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
    print(f"[window {window_idx}] {summary}")


def main():
    df = pd.read_csv(os.path.join("data", "stream.csv"))
    texts = df["text"].tolist()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"Streaming {len(texts)} headlines to {API_URL} "
          f"(window size {WINDOW_SIZE})")

    run_name = f"monitoring-{time.strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "window_size": WINDOW_SIZE,
                "n_texts": len(texts),
                "api_url": API_URL,
                "sleep_ms": SLEEP_MS,
            }
        )

        window: list[dict] = []
        window_idx = 0
        failures = 0

        for i, text in enumerate(texts, start=1):
            try:
                window.append(predict(text))
            except Exception as e:  # noqa: BLE001 - one bad row must not end the stream
                failures += 1
                print(f"[WARN] row {i} failed: {e}")
                continue

            time.sleep(SLEEP_MS / 1000.0)

            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window_idx += 1
                window = []

        # Flush the trailing partial window so its samples are not dropped.
        if window:
            log_window(window, window_idx)
            window_idx += 1

        mlflow.log_metric("failed_requests", failures)
        print(f"\nDone: {window_idx} windows logged, {failures} failed requests.")


if __name__ == "__main__":
    main()
