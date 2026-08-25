"""
Load test for the Sentiment Analysis API.

Run: locust -f scripts/locustfile.py --host http://localhost:8067

Then open http://localhost:8967 to configure and start the test.
"""

import random

from locust import HttpUser, between, task

SAMPLE_HEADLINES = [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
    "Stocks closed flat on Friday amid low trading volume.",
    "Federal Reserve signals interest rate cuts later this year.",
    "Tech giant announces major layoffs amid revenue decline.",
    "Merger talks between the two firms collapsed overnight.",
    "Quarterly earnings beat analyst expectations by wide margin.",
    "Oil prices surge on supply concerns from the Middle East.",
]


class SentimentAPIUser(HttpUser):
    """Simulates a client mixing single predictions, batches, and health polls."""

    # Think time between requests, so we model paced users rather than a tight loop.
    wait_time = between(0.5, 2.0)

    @task(3)
    def predict_single(self):
        self.client.post(
            "/predict",
            json={"text": random.choice(SAMPLE_HEADLINES)},
            name="POST /predict",
        )

    @task(1)
    def predict_batch(self):
        self.client.post(
            "/predict/batch",
            json={"texts": random.sample(SAMPLE_HEADLINES, 4)},
            name="POST /predict/batch",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="GET /health")
