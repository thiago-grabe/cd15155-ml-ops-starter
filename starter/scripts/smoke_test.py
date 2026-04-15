"""
Smoke test to confirm the FinBERT model artifact loads correctly
and runs a basic inference.

Run:
    python starter/scripts/smoke_test.py
"""

import sys


def check_model():
    print("Loading ProsusAI/finbert model and tokenizer...")
    try:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
        result = classifier("The company reported record profits this quarter.")
        assert result and result[0]["label"] in {"positive", "negative", "neutral"}
        print(f"Model loaded and inference OK  →  {result[0]}")
        return True
    except Exception as e:
        print(f"[FAIL] Model check failed: {e}")
        return False


def main():
    print("=" * 50)
    print("  FinBERT Smoke Test")
    print("=" * 50)

    passed = check_model()

    print("=" * 50)
    if passed:
        print("  All checks passed.")
        sys.exit(0)
    else:
        print("  Smoke test FAILED. Fix the errors above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
