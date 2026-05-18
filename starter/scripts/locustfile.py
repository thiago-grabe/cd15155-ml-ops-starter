"""
Load test for the Sentiment Analysis API.

Run: locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure and start the test.
"""

from locust import HttpUser

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
    # TODO: Implement three load test tasks using the @task decorator:
    # 1. predict_single (weight 3): POST a randomly chosen headline to /predict
    # 2. predict_batch (weight 1): POST a random sample of 4 headlines to /predict/batch
    # 3. health_check (weight 1): GET /health
    # Hint: set wait_time to simulate realistic user pacing.
    # Documentation: https://docs.locust.io/en/stable/writing-a-locustfile.html
    pass
