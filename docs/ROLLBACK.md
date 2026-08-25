# Model Rollback

Reverts the MLflow `production` alias to the previous registered version, so the last
known-good model resumes serving. This is the remediation path when the Deepchecks drift
gate fails on a newly promoted model.

## When to use it

The drift gate (`scripts/run_deepchecks.py`) exits non-zero when property drift or
prediction drift exceeds the `params.yaml` thresholds, which blocks `build` and `deploy`.
That protects *new* deploys, but if a bad model was already promoted to `@production`,
the running API keeps loading it. Rollback moves the alias back.

## Trigger it

**Locally** (the tracking server runs in compose):

```bash
cd starter
docker compose --profile tools run --rm tooling python scripts/rollback.py --dry-run
docker compose --profile tools run --rm tooling python scripts/rollback.py
docker compose restart api      # reload the aliased model
```

**Via GitHub Actions**:

```bash
gh workflow run rollback.yml -f dry_run=true
gh workflow run rollback.yml -f dry_run=false
gh workflow run rollback.yml -f version=3 -f dry_run=false   # specific version
```

Or: Actions → *Model rollback* → *Run workflow*.

> The workflow needs an `MLFLOW_TRACKING_URI` secret pointing at a tracking server the
> runner can actually reach. An MLflow instance running only in local docker-compose is
> **not** reachable from a GitHub-hosted runner — use the local command in that case.
> The job also inherits the `production` environment, so it respects the same approval gate
> as deploy.

## Behaviour

| Situation | Result |
|---|---|
| `@production` on v4, v3 exists | alias moves to v3 |
| `--to N` given | alias moves to vN (error if N is not registered) |
| Already on the target | no-op, exits 0 |
| No older version exists | **exits 1** — refuses to roll back |
| `--dry-run` | reports the intended change, applies nothing |

The "no older version" case fails loudly on purpose: there is nothing safe to fall back
to, and the drift failure needs investigating rather than papering over.

Defaults are conservative — the workflow's `dry_run` input defaults to `true`, so an
accidental trigger reports instead of acting.
