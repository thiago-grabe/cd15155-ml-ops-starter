"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from app.utils import load_classifier
from dotenv import load_dotenv

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    # TODO: Create two TextData objects from the text lists above: one for stream_texts
    # and one for test_texts. Use task_type="text_classification".
    # Then call calculate_builtin_properties() on each to compute NLP properties.
    # https://docs.deepchecks.com/stable/nlp/usage_guides/text_data_object.html#nlp-textdata-object

    # TODO: Compute NLP Property Drift between test_dataset (reference) and stream_dataset (current).
    # Extract the "Sentiment" drift score from the result and compare it against
    # property_drift_threshold.
    # If it exceeds the threshold, print an error message and terminate with failure exit code
    # https://docs.deepchecks.com/stable/nlp/auto_checks/train_test_validation/plot_property_drift.html#nlp-property-drift

    # TODO: Run predictions on both datasets with run_predictions(), then compute Prediction Drift.
    # Add a condition to fail if drift score is above the prediction_drift_threshold
    # If drift exceeds threshold, print an error message and terminate with failure exit code
    # https://docs.deepchecks.com/stable/general/guides/drift_guide.html#text-nlp-checks


if __name__ == "__main__":
    main()
