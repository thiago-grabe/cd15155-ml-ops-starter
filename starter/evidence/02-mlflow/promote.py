"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

import os
import sys

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

PRODUCTION_ALIAS = "production"


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found. Run scripts/evaluate.py first."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished runs in experiment '{experiment_name}'. "
            "Run scripts/evaluate.py first."
        )

    run = runs[0]
    f1 = run.data.metrics.get("f1_weighted")
    if f1 is None:
        raise RuntimeError(
            f"Run {run.info.run_id} has no 'f1_weighted' metric; cannot evaluate promotion."
        )
    return run.info.run_id, float(f1)


def promote(client: MlflowClient, model_name: str) -> str:
    """Assign the production alias to the highest-numbered registered version."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions of '{model_name}'. Run scripts/evaluate.py first."
        )

    # Pick by version NUMBER, not by list order: search_model_versions sorts by
    # last_updated_timestamp DESC, and setting an alias bumps that timestamp, which
    # makes the default ordering unstable across repeated runs.
    latest = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, latest.version)
    print(f"PROMOTED: {model_name} v{latest.version} -> @{PRODUCTION_ALIAS}")

    # app/utils.py resolves models:/{MODEL_NAME}@{MODEL_STAGE}. If MODEL_STAGE is set to
    # something other than "production", mirror the alias so the API can still load it.
    stage_alias = os.getenv("MODEL_STAGE", PRODUCTION_ALIAS)
    if stage_alias and stage_alias != PRODUCTION_ALIAS:
        client.set_registered_model_alias(model_name, stage_alias, latest.version)
        print(f"          also aliased @{stage_alias} (MODEL_STAGE)")

    return latest.version


def main():
    threshold = load_params()["f1_threshold"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    model_name = os.getenv("MODEL_NAME", "finbert")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run_id, f1 = get_latest_f1(client, experiment_name)
    print(f"Latest evaluation run: {run_id}")
    print(f"  f1_weighted = {f1:.4f}")
    print(f"  threshold   = {threshold}")

    if f1 >= threshold:
        promote(client, model_name)
    else:
        print(
            f"\nNOT PROMOTED: f1_weighted {f1:.4f} is below the required "
            f"threshold {threshold}. The production alias was left unchanged."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
