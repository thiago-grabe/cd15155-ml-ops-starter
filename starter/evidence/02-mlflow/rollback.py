"""
Roll the MLflow `production` alias back to the previous registered model version.

Intended as the remediation step when the Deepchecks drift gate fails: the newly
promoted model is demoted and the last known-good version resumes serving.

Run:
    python scripts/rollback.py              # roll back one version
    python scripts/rollback.py --to 3       # roll back to a specific version
    python scripts/rollback.py --dry-run    # show what would happen
"""

import argparse
import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

PRODUCTION_ALIAS = "production"


def parse_args():
    p = argparse.ArgumentParser(description="Roll back the production model alias.")
    p.add_argument("--to", type=int, default=None,
                   help="Explicit version to roll back to (default: the previous one).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the intended change without applying it.")
    return p.parse_args()


def current_production_version(client: MlflowClient, model_name: str) -> int | None:
    try:
        mv = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        return int(mv.version)
    except Exception:  # noqa: BLE001 - "no alias set yet" is a normal, expected state
        return None


def main():
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "finbert")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    versions = sorted(
        (int(v.version) for v in client.search_model_versions(f"name='{model_name}'")),
    )
    if not versions:
        print(f"ERROR: no registered versions of '{model_name}'.")
        sys.exit(1)

    current = current_production_version(client, model_name)
    print(f"Model            : {model_name}")
    print(f"Known versions   : {versions}")
    print(f"Current @{PRODUCTION_ALIAS} : {current if current is not None else '(unset)'}")

    if args.to is not None:
        target = args.to
        if target not in versions:
            print(f"ERROR: version {target} is not registered for '{model_name}'.")
            sys.exit(1)
    else:
        candidates = [v for v in versions if current is None or v < current]
        if not candidates:
            print(
                f"\nERROR: no version older than v{current} exists — nothing to roll "
                "back to. Investigate the drift failure instead of rolling back."
            )
            sys.exit(1)
        target = candidates[-1]

    if current == target:
        print(f"\nAlready serving v{target}; nothing to do.")
        return

    if args.dry_run:
        print(f"\nDRY RUN: would move @{PRODUCTION_ALIAS} from v{current} to v{target}.")
        return

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, str(target))
    print(f"\nROLLED BACK: @{PRODUCTION_ALIAS} {current} -> {target}")
    print("Restart the API (or redeploy) so it reloads the aliased model.")


if __name__ == "__main__":
    main()
