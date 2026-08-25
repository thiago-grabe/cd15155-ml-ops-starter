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
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

from app.utils import load_classifier

load_dotenv()

# Only the Sentiment property is computed, and English-sample filtering is switched off.
#
# Both settings are for SPEED, not for dependency avoidance. It is tempting to assume they
# let you drop fasttext -- they do not. deepchecks 0.19.1 calls
#     kwargs['fasttext_model'] = get_fasttext_model(...)
# unconditionally in calculate_builtin_properties(), BEFORE any property filtering is
# applied, so fasttext is imported no matter which properties you ask for. It is therefore
# a hard dependency (built from sdist in docker/Dockerfile.tooling).
#
# What these kwargs do buy:
#   include_properties=["Sentiment"]  -> skips ~20 other properties, several of which pull
#                                        ONNX transformer models (Toxicity, Fluency, ...).
#   ignore_non_english...=False       -> skips per-batch language detection over every row.
#
# "Sentiment" is TextBlob-based and is the property the drift gate is specified against.
DRIFT_PROPERTY = "Sentiment"
MIN_SAMPLES = 100  # deepchecks' own default; below this the checks raise instead of scoring


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32, truncation=True, max_length=512)
    return [r["label"] for r in results]


def _drift_score(value) -> float:
    """Normalise a deepchecks result payload to a single float.

    The shape differs by check: PropertyDrift gives
    {<property>: {'Drift score': ..., 'Method': ..., 'Importance': ...}} while
    PredictionDrift gives {'Drift score': ..., 'Method': ..., 'Samples per class': ...}.
    """
    if isinstance(value, dict):
        for key in ("Drift score", "drift_score", "Drift Score"):
            if key in value:
                inner = value[key]
                return _drift_score(inner) if isinstance(inner, dict) else float(inner)
        # A per-class mapping: report the worst case.
        numeric = [v for v in value.values() if isinstance(v, (int, float))]
        if numeric:
            return float(max(numeric))
    return float(value)


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

    print(f"Reference (test.csv):  {len(test_texts)} samples")
    print(f"Current   (stream.csv): {len(stream_texts)} samples")
    for label, texts in (("test.csv", test_texts), ("stream.csv", stream_texts)):
        if len(texts) < MIN_SAMPLES:
            print(
                f"ERROR: {label} has {len(texts)} rows, below deepchecks' "
                f"min_samples={MIN_SAMPLES}. The drift checks cannot produce a score.\n"
                "       Relax clean.min_words / clean.max_words in params.yaml, or "
                "supply more raw data, rather than lowering min_samples."
            )
            sys.exit(1)

    # test is the REFERENCE distribution, stream is the CURRENT one.
    stream_dataset = TextData(stream_texts, task_type="text_classification")
    test_dataset = TextData(test_texts, task_type="text_classification")

    print(f"\nCalculating '{DRIFT_PROPERTY}' property...")
    for dataset in (test_dataset, stream_dataset):
        dataset.calculate_builtin_properties(
            include_properties=[DRIFT_PROPERTY],
            ignore_non_english_samples_for_english_properties=False,
        )

    # ---- NLP property drift -------------------------------------------------
    property_result = PropertyDrift(
        # Sentiment has output_type 'numeric', so the NUMERIC threshold is the one that
        # applies; max_allowed_categorical_score would silently never fire for it.
        min_samples=MIN_SAMPLES
    ).add_condition_drift_score_less_than(
        max_allowed_numeric_score=property_drift_threshold
    ).run(train_dataset=test_dataset, test_dataset=stream_dataset)

    property_score = _drift_score(property_result.value[DRIFT_PROPERTY])
    print(
        f"\n{DRIFT_PROPERTY} property drift: {property_score:.4f} "
        f"(max allowed {property_drift_threshold})"
    )
    if property_score > property_drift_threshold:
        print(
            f"FAIL: {DRIFT_PROPERTY} property drift {property_score:.4f} exceeds "
            f"threshold {property_drift_threshold}."
        )
        sys.exit(1)

    # ---- Prediction drift ---------------------------------------------------
    print("\nRunning predictions for prediction drift...")
    test_predictions = run_predictions(classifier, test_texts)
    stream_predictions = run_predictions(classifier, stream_texts)

    prediction_result = (
        PredictionDrift()
        # NOTE: PredictionDrift takes max_allowed_drift_score. Passing PropertyDrift's
        # max_allowed_categorical_score here raises TypeError.
        .add_condition_drift_score_less_than(
            max_allowed_drift_score=prediction_drift_threshold
        )
        .run(
            train_dataset=test_dataset,
            test_dataset=stream_dataset,
            train_predictions=test_predictions,
            test_predictions=stream_predictions,
        )
    )

    prediction_score = _drift_score(prediction_result.value)
    print(
        f"Prediction drift: {prediction_score:.4f} "
        f"(max allowed {prediction_drift_threshold})"
    )

    failed = [c for c in prediction_result.conditions_results if not c.is_pass()]
    if prediction_score > prediction_drift_threshold or failed:
        print(
            f"FAIL: prediction drift {prediction_score:.4f} exceeds threshold "
            f"{prediction_drift_threshold}."
        )
        sys.exit(1)

    print("\nDeepchecks passed: no drift above threshold.")


if __name__ == "__main__":
    main()
