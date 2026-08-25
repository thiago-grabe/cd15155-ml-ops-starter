# Setup and Runbook

Everything runs in containers. Nothing heavy is installed on the host — no Python
virtualenv, no torch, no DVC. The only host prerequisites are **Docker Desktop** and
(optionally) `gh` and `aws` for the CI/CD and AWS steps.

> Local Python here is 3.14, for which torch publishes no wheels, so a host install is
> not merely undesirable — it is not possible. The containers use Python 3.12.

## Ports

Every published host port carries a `67` suffix to avoid clashing with other local apps.
Container ports stay canonical, because `prometheus.yml` and the ECS task definition both
target the container port.

| Service | URL | Container port |
|---|---|---|
| API | http://localhost:8067 | 8000 |
| MLflow | http://localhost:5067 | 5000 |
| Prometheus | http://localhost:9067 | 9090 |
| Grafana | http://localhost:3067 (admin/admin) | 3000 |
| Locust | http://localhost:8967 | 8089 |

MLflow is on 5067 because host port 5000 is occupied by macOS ControlCenter (AirPlay
Receiver). To free it: System Settings → General → AirDrop & Handoff → AirPlay Receiver off.

> **Note on `8067:8000`.** The project brief names an `8000:8000` mapping. It is remapped
> here for host-port consistency; the container still listens on 8000 and the original
> mapping is preserved as a comment in `docker-compose.yml`. Reverting is a one-line change.

## Behind a TLS-inspecting proxy

This machine sits behind Zscaler, which re-signs `download.pytorch.org` and
`huggingface.co`. Containers do not trust that root, so image builds fail with
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.

Extract the proxy root CA once — the Dockerfiles pick up anything in `docker/certs/*.crt`:

```bash
security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain \
  > starter/docker/certs/zscaler-root.crt
```

The file is git-ignored (it is network-specific and not ours to publish), and the
directory is empty on GitHub Actions, where the extra trust is a no-op.

## First run, in order

Dependencies between these steps are real — the order matters.

```bash
cd starter
cp .env.example .env          # defaults are already coherent
mkdir -p mlruns mlartifacts

# 1. Build all images
docker compose --profile tools build

# 2. Start the MLflow tracking server
docker compose up -d mlflow

# 3. Download + patch the model config (fixes float id2label upstream).
#    Writes into the shared hf-cache volume, so the api container reuses it.
docker compose --profile tools run --rm tooling python scripts/smoke_test.py

# 4. Data pipeline -> data/train.csv, data/test.csv, data/stream.csv + dvc.lock
docker compose --profile tools run --rm tooling dvc init --subdir
docker compose --profile tools run --rm tooling dvc repro

# 5. Evaluate and register the model
docker compose --profile tools run --rm tooling python scripts/evaluate.py

# 6. Promote to the @production alias (only if F1 >= params.yaml threshold)
docker compose --profile tools run --rm tooling python scripts/promote.py

# 7. Now the registry is populated, so MODEL_SOURCE=mlflow is safe.
sed -i '' 's/^MODEL_SOURCE=huggingface/MODEL_SOURCE=mlflow/' .env
docker compose up -d --build api prometheus grafana
```

`MODEL_SOURCE` starts as `huggingface` deliberately: with `mlflow` and an empty registry
the API lifespan fails and Compose restart-loops.

## Verification

```bash
docker compose --profile tools run --rm tooling pytest tests/ -v
docker compose --profile tools run --rm tooling python scripts/run_deepchecks.py
docker compose --profile tools run --rm tooling python monitoring/stream.py
docker compose --profile tools run --rm tooling \
  locust -f scripts/locustfile.py --host http://api:8000
```

Then regenerate the full rubric proof pack:

```bash
./evidence/collect_evidence.sh
open evidence/README.md
```

## Image size (stand-out: slim production image)

The API image installs a CPU-only torch wheel from `download.pytorch.org/whl/cpu` in its
own layer, and uses `requirements-api.txt` rather than the full toolchain.

`--index-url` **replaces** PyPI rather than adding to it (that index 404s for
fastapi/pandas/transformers), so the two installs cannot be merged into one command.

| Build | Approx. size | Why |
|---|---|---|
| Default PyPI torch | ~6–7 GB | 888 MB torch wheel + ~15 `nvidia-*`/`triton` CUDA packages |
| CPU-only wheel + `requirements-api.txt` | ~1.2 GB | 184 MB torch wheel, no CUDA, no dvc/deepchecks/locust/pytest |

Measured values for this build are in `evidence/04-docker/image-sizes.txt`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CERTIFICATE_VERIFY_FAILED` during build | Zscaler TLS interception | Extract the proxy CA (above) |
| api container stuck `starting` | Cold model load | `start_period` is 180s; wait, then `docker compose logs api` |
| api restart-loops on boot | `MODEL_SOURCE=mlflow` with an empty registry | Set `MODEL_SOURCE=huggingface`, run evaluate + promote |
| `NotEnoughSamplesError` from deepchecks | `data/stream.csv` under 100 rows | Relax `clean.min_words`/`max_words` in `params.yaml` — do not lower `min_samples` |
| `dubious ownership` from git in the tooling container | Bind-mounted repo owned by another uid | Already handled via `safe.directory` in `Dockerfile.tooling` |
