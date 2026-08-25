# AWS Setup

Creates the resources the CI/CD pipeline deploys into: an ECR repository, a security
group, a CloudWatch log group, an ECS execution role, an ECS cluster, a task definition,
and the ECS service.

## Account context — read this first

The credentials for this project are **temporary Vocareum lab credentials**:

- Identity: `arn:aws:sts::402398991984:assumed-role/voclabs/user…`
- Access key starts with `ASIA…` and **requires a session token**
- **They expire** (typically within a few hours, and always when the lab session ends)

Two consequences that will bite if ignored:

1. `AWS_SESSION_TOKEN` must be set as a GitHub secret alongside the key id and secret.
   Without it every AWS call in CI fails with `InvalidClientTokenId`.
2. All three secrets must be **re-set after every lab refresh**. A pipeline that worked
   yesterday will fail today with no code change.

> Credentials pasted into a terminal end up in shell history and logs. Treat any
> credential that has been pasted as compromised: let it expire, don't reuse it, and
> never commit it. `.env` is git-ignored.

## Known constraint: no ECS execution role

Fargate **requires** an `executionRoleArn` to pull from ECR and write to CloudWatch Logs.
This account has none, and cannot be pre-checked:

```
$ aws iam list-roles --query 'Roles[].RoleName'
# service-linked roles only, plus: vocareum, vocareum-eventbridge, voclabs
# -> no LabRole, no ecsTaskExecutionRole

$ aws iam simulate-principal-policy ...
# AccessDenied — we cannot test permissions without exercising them
```

`scripts/aws_bootstrap.sh` therefore probes at runtime, in order:

1. Use `ecsTaskExecutionRole` if it exists.
2. Otherwise try to create it with an `ecs-tasks.amazonaws.com` trust policy and attach
   `AmazonECSTaskExecutionRolePolicy`.
3. If `iam:CreateRole` is denied, fall back to `LabRole` → `voclabs` → `vocareum`,
   **verifying each actually trusts `ecs-tasks.amazonaws.com`** before using it (an
   untrusted role fails later with a confusing "ECS was unable to assume the role").
4. If none works, it stops with an explicit message rather than registering a task
   definition that can never start.

**If step 4 is reached** — time-box further attempts to ~30 minutes, then:

- Restart the Vocareum lab session; some provision a `LabRole` on reset.
- Or switch to the ECS **EC2** launch type, which pulls via the instance profile and
  needs no execution role.
- Or demonstrate the service via `docker compose up` and record the IAM restriction.
  The graded artifacts are the workflow file and the bootstrap script, not a live URL.

Watch for `iam:PassRole` being denied separately — it blocks `RegisterTaskDefinition`
even when the role exists.

## Run it

```bash
export AWS_ACCESS_KEY_ID="…"
export AWS_SECRET_ACCESS_KEY="…"
export AWS_SESSION_TOKEN="…"
export AWS_REGION="us-east-1"

cd starter
./scripts/aws_bootstrap.sh
```

Creates, idempotently:

| Resource | Name | Notes |
|---|---|---|
| ECR repository | `finbert-api` | scan-on-push enabled |
| Security group | `finbert-sg` | inbound **80** (per brief) **and 8000** |
| Log group | `/ecs/finbert-api` | pre-created, so the task needs no `logs:CreateLogGroup` |
| ECS cluster | `finbert-cluster` | |
| Task definition | `finbert-api` | Fargate, awsvpc, 1 vCPU / 3 GB, X86_64 |
| ECS service | `finbert-api-service` | `assignPublicIp=ENABLED` |

**Why port 8000 must be open**: with `awsvpc` networking the security group attaches to
the task ENI and there is no port translation. Opening only 80 gives a task that reports
healthy but is unreachable.

**Why public subnets**: this account has no NAT gateway, so a task in a private subnet
cannot reach ECR and fails with `CannotPullContainerError`.

**Why 3 GB**: torch plus FinBERT does not fit comfortably in 2 GB. `1024/3072` is a valid
Fargate cpu/memory pair.

## GitHub secrets

```bash
gh secret set AWS_ACCESS_KEY_ID     --body "$AWS_ACCESS_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY --body "$AWS_SECRET_ACCESS_KEY"
gh secret set AWS_SESSION_TOKEN     --body "$AWS_SESSION_TOKEN"
gh secret set AWS_REGION            --body "us-east-1"
gh secret set ECR_REGISTRY          --body "402398991984.dkr.ecr.us-east-1.amazonaws.com"
```

### Credential refresh runbook

When the pipeline starts failing with `ExpiredToken` or `InvalidClientTokenId`:

1. Reopen the Vocareum lab → **AWS Details → AWS CLI** → copy the new triple.
2. Re-export them locally and re-run the three `gh secret set` commands above.
3. Re-run `./scripts/aws_bootstrap.sh` (idempotent) to confirm access.
4. Re-run the failed workflow: `gh run rerun <run-id>`.

## Architecture note

The task definition targets **X86_64**. GitHub runners are natively amd64, so let CI build
the deployed image. Building on an Apple Silicon Mac produces an arm64 manifest that ECS
cannot run — a local push must use:

```bash
docker buildx build --platform linux/amd64 -t <registry>/finbert-api:latest --push .
```

## Deploy approval gate (stand-out)

The `deploy` job declares `environment: production`. To make it an actual gate:
**Settings → Environments → production → Required reviewers**. The job then pauses for
manual approval before touching ECS.
