# Building a Real-Time Financial News Sentiment Service

Operationalise a pre-trained financial news sentiment classifier by building the full MLOps stack: data versioning, model registry, REST API, containerisation, CI/CD, and production monitoring.

## This implementation runs entirely in containers

Nothing heavy is installed on the host — no virtualenv, no torch, no DVC. A `tooling`
container runs every script; the stack itself (API, MLflow, Prometheus, Grafana) is
docker-compose.

**Start here: [docs/SETUP.md](docs/SETUP.md)** for the ordered runbook,
[docs/AWS_SETUP.md](docs/AWS_SETUP.md) for the deployment, and
[docs/ROLLBACK.md](docs/ROLLBACK.md) for the model rollback procedure.

Machine-generated, re-runnable proof that each rubric item is met lives in
[starter/evidence/](starter/evidence/):

```bash
cd starter && ./evidence/collect_evidence.sh   # regenerates evidence/README.md
```

### Ports

Published host ports carry a `67` suffix to avoid collisions with other local apps.
Container ports remain canonical (8000/5000/9090/3000).

| Service | URL |
|---|---|
| API | http://localhost:8067 |
| MLflow | http://localhost:5067 |
| Prometheus | http://localhost:9067 |
| Grafana | http://localhost:3067 |
| Locust | http://localhost:8967 |

### Quick start

```bash
cd starter
cp .env.example .env
docker compose --profile tools build
docker compose up -d mlflow
docker compose --profile tools run --rm tooling python scripts/smoke_test.py
docker compose --profile tools run --rm tooling dvc init --subdir
docker compose --profile tools run --rm tooling dvc repro
docker compose --profile tools run --rm tooling python scripts/evaluate.py
docker compose --profile tools run --rm tooling python scripts/promote.py
# then set MODEL_SOURCE=mlflow in .env
docker compose up -d --build api prometheus grafana
```

---

## Learning objectives

By the end of this project, you will be able to:

- Load and version datasets using DVC

- Evaluate and register a model in MLflow

- Write a prediction API 

- Package and deploy the prediction service using Docker and Docker-compose

- Write a CI/CD pipeline that pushes the image to ECR and deploys the service in AWS

- Collect service metrics using Prometheus

- Add production monitoring using MLflow


## Prerequisites

- Python 3.12
- An AWS account with ECR access
- Git

## Getting Started

### 1. Clone the repo

```bash
git clone
```

> All the dependencies are installed in the workspace. Please refrain from installing dependencies using pip

Navigate to the project root 
```bash
cd starter
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in the required values in `.env`.

### 3. Download and patch the model, then validate your setup

```bash
python scripts/smoke_test.py
```

### 4. Launch Mlflow server
```
docker compose up -d mlflow   # http://localhost:5067
```

## Project Structure

```
starter/
├── app/
│   ├── main.py              # FastAPI service (Tasks 3 and 6)
│   └── utils.py             # Model loading 
├── data/                    # DVC-tracked datasets
│   ├── raw_stream.csv       # Raw Bluesky financial posts
├── monitoring/
│   └── stream.py            # MLflow monitoring loop (Task 6)
├── scripts/
│   ├── load_data.py         # Downloads financial dataset
│   ├── clean_data.py        # Cleans raw Bluesky stream
│   ├── evaluate.py          # Evaluates model on test set, logs to MLflow (Task 2)
│   ├── promote.py           # Promotes model to production alias if F1 >= threshold (Task 2)
│   ├── run_deepchecks.py    # CI quality gate: property drift + prediction drift (Task 5)
│   └── locustfile.py        # Locust load testing (Task 3)
├── tests/
│   └── test_api.py          # pytest integration tests for the API (Task 3)
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # CI/CD pipeline (Task 5)
├── dvc.yaml                 # DVC pipeline definition (Task 1)
├── params.yaml              # Configuration parameters 
├── docker-compose.yml       # Local API service orchestration (Task 4)
├── Dockerfile               # Production image (Task 4)
├── prometheus.yml           # Prometheus scrape configuration (Task 6)
├── requirements.txt         # All dependencies
└── .env.example             # Environment variable template
```

## Project Tasks

### Task 0 — Project Setup
Configure your environment and confirm the model loads correctly.

### Task 1 — Data Versioning (DVC)
Complete [dvc.yaml](starter/dvc.yaml) to define the data preparation pipeline, run `dvc repro`.

```bash
docker compose --profile tools run --rm tooling dvc init --subdir
docker compose --profile tools run --rm tooling dvc repro
```

**Deliverable:** [dvc.yaml](starter/dvc.yaml), `dvc.lock`

### Task 2 — Model Evaluation and Registration (MLflow)
Complete [scripts/evaluate.py](starter/scripts/evaluate.py) to track the evaluation run with MLflow: set the tracking URI and experiment name, log model parameters and metrics, then log and register the pipeline in the Model Registry. Then complete [scripts/promote.py](starter/scripts/promote.py) to automatically promote the model to Production if it meets the F1 threshold.

```bash
docker compose --profile tools run --rm tooling python scripts/evaluate.py
docker compose --profile tools run --rm tooling python scripts/promote.py
```

**Deliverable:** [scripts/evaluate.py](starter/scripts/evaluate.py), model registered in MLflow, [scripts/promote.py](starter/scripts/promote.py)

### Task 3 — Prediction API (FastAPI)
Complete [app/main.py](starter/app/main.py) to implement three endpoints:
- `POST /predict` — single headline → `{sentiment, confidence, latency_ms}`
- `POST /predict/batch` — list of headlines → list of above
- `GET /health` — service health status

Write integration tests in [tests/test_api.py](starter/tests/test_api.py). Complete [scripts/locustfile.py](starter/scripts/locustfile.py) to simulate concurrent users hitting the API endpoints under load.

```bash
docker compose up -d api
docker compose --profile tools run --rm tooling pytest tests/ -v
docker compose --profile tools run --rm tooling \
  locust -f scripts/locustfile.py --host http://api:8000
```

**Deliverable:** [app/main.py](starter/app/main.py), [tests/test_api.py](starter/tests/test_api.py), [scripts/locustfile.py](starter/scripts/locustfile.py)

### Task 4 — Containerisation
Complete [Dockerfile](starter/Dockerfile) and [docker-compose.yml](starter/docker-compose.yml) to run the FastAPI service with proper port mapping, environment variables, volume mounts for MLflow artifacts, and a health check.

> The original brief says not to build locally (a Udacity workspace constraint). This
> implementation is container-first by design, so images are built locally on purpose. 

**Deliverable:** [Dockerfile](starter/Dockerfile), [docker-compose.yml](starter/docker-compose.yml)

### Task 5 — CI/CD Pipeline
Complete [scripts/run_deepchecks.py](starter/scripts/run_deepchecks.py) to run property drift and prediction drift checks. 
Then complete [.github/workflows/ci-cd.yml](starter/.github/workflows/ci-cd.yml) to build a pipeline that on every push to `main`:
1. Runs integration tests
2. Runs Deepchecks model quality checks
3. Builds and pushes the Docker image to Amazon ECR
4. Deploys the service on AWS

```bash
docker compose --profile tools run --rm tooling python scripts/run_deepchecks.py
```

#### AWS Setup

Before the pipeline can deploy, create the following AWS resources:

1. **ECR repository** named `finbert-api` to store Docker images
2. **Security group** named `finbert-sg` — inbound HTTP on port 80 from 0.0.0.0/0
2. **ECS cluster** named `finbert-cluster`
3. **ECS task definition** named `finbert-api` configured to run the container
4. **ECS service** named `finbert-api-service` on the cluster

> You can follow this [guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started-fargate.html#first-run-linux-prereqs)

Then add these secrets to your GitHub repository 

```bash
gh secret set AWS_ACCESS_KEY_ID --body "..."
gh secret set AWS_SECRET_ACCESS_KEY --body "..."
gh secret set AWS_REGION --body "us-east-1"
gh secret set ECR_REGISTRY --body ""
```

**Deliverable:** [scripts/run_deepchecks.py](starter/scripts/run_deepchecks.py), [.github/workflows/ci-cd.yml](starter/.github/workflows/ci-cd.yml)

### Task 6 — Monitoring
Add structured JSON logging and instrument [app/main.py](starter/app/main.py) with Prometheus counters and histograms. Configure [prometheus.yml](starter/prometheus.yml) to scrape the `/metrics` endpoint. Then complete [monitoring/stream.py](starter/monitoring/stream.py) to simulate production traffic and log aggregated metrics (sentiment distribution, confidence, latency) to MLflow.

```bash
docker compose --profile tools run --rm tooling python monitoring/stream.py
```

**Deliverable:** [app/main.py](starter/app/main.py), [prometheus.yml](starter/prometheus.yml), [monitoring/stream.py](starter/monitoring/stream.py)


## Resources

- [FinBERT Market Based](https://huggingface.co/baptle/FinBERT_market_based) — pre-trained financial sentiment model
- [Financial Headlines Market Based](https://huggingface.co/datasets/baptle/financial_headlines_market_based) — financial news headlines dataset
- [HuggingFace Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) — model loading and inference pipeline
- [DVC](https://dvc.org/) — data version control
- [MLflow](https://mlflow.org/) — experiment tracking and model registry
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [Docker](https://www.docker.com/) — containerisation
- [Prometheus](https://prometheus.io/)  — monitoring
- [Deepchecks](https://deepchecks.com/) — model quality checks
- [GitHub Actions](https://docs.github.com/en/actions) — CI/CD
- [Amazon ECR](https://docs.aws.amazon.com/ecr/) — container registry
- [AWS Fargate on ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started-fargate.html) — serverless container deployment

## License

[License](../LICENSE.txt)