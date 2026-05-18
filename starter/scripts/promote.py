"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""
# TODO: Write a script that :
# 1- reads the F1 threshold from params.yaml,
# 2- fetches the weighted F1 score from the most recent MLflow evaluation run,
# 3- promotes the latest model version to a "production" alias if the score
# meets the threshold.
# 4- Print a clear message when the model does not qualify.
# Documentation: https://mlflow.org/docs/latest/model-registry.html

import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    # TODO: Look up the experiment by name, search for the most recent run,
    # and return its run_id and f1_weighted metric value.
    # Documentation: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html
    raise NotImplementedError


def promote(client: MlflowClient, model_name: str):
    # TODO: Fetch all registered versions of the model, identify the latest one,
    # and assign it the "production" alias using the MLflow client.
    # Documentation: https://mlflow.org/docs/latest/ml/model-registry/

    # get the latest version of the model
    raise NotImplementedError


def main():
    # TODO: Write the main function:
    # 1- load the threshold from params.yaml,
    # 2- call get_latest_f1(),
    # 3- call promote() only if the threshold is met.
    raise NotImplementedError


if __name__ == "__main__":
    main()
