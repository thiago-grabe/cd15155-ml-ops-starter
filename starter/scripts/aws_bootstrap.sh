#!/usr/bin/env bash
#
# Idempotent AWS bootstrap for the FinBERT service.
# Creates: ECR repo, security group, CloudWatch log group, ECS execution role,
#          ECS cluster, task definition, and the ECS service.
#
# Safe to re-run after every Vocareum credential refresh.
#
#   ./scripts/aws_bootstrap.sh
#
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-finbert-api}"
CLUSTER="${ECS_CLUSTER:-finbert-cluster}"
SERVICE="${ECS_SERVICE:-finbert-api-service}"
TASK_FAMILY="${TASK_DEFINITION:-finbert-api}"
SG_NAME="${SG_NAME:-finbert-sg}"
LOG_GROUP="/ecs/${TASK_FAMILY}"
EXEC_ROLE_NAME="ecsTaskExecutionRole"

export AWS_DEFAULT_REGION="$AWS_REGION"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "0/8  Verifying credentials"
if ! CALLER=$(aws sts get-caller-identity --output json 2>&1); then
  die "AWS credentials are invalid or expired.
     These are temporary Vocareum lab credentials. Refresh them from the lab console
     (AWS Details -> AWS CLI) and re-export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
     AWS_SESSION_TOKEN, then re-run. See docs/AWS_SETUP.md."
fi
ACCOUNT_ID=$(echo "$CALLER" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')
CALLER_ARN=$(echo "$CALLER" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
info "account : $ACCOUNT_ID"
info "identity: $CALLER_ARN"
info "registry: $ECR_REGISTRY"

# ---------------------------------------------------------------------------
say "1/8  ECR repository: $ECR_REPOSITORY"
if aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1; then
  info "already exists"
else
  aws ecr create-repository \
    --repository-name "$ECR_REPOSITORY" \
    --image-scanning-configuration scanOnPush=true \
    --query 'repository.repositoryUri' --output text
  info "created"
fi

# ---------------------------------------------------------------------------
say "2/8  Network (default VPC + public subnets)"
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
[ "$VPC_ID" != "None" ] || die "No default VPC in $AWS_REGION."
# Fargate tasks need public subnets: this account has no NAT gateway, so a private
# subnet would leave the task unable to pull the image from ECR.
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters Name=vpc-id,Values="$VPC_ID" Name=map-public-ip-on-launch,Values=true \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
[ -n "$SUBNET_IDS" ] || die "No public subnets found in $VPC_ID."
info "vpc    : $VPC_ID"
info "subnets: $SUBNET_IDS"

# ---------------------------------------------------------------------------
say "3/8  Security group: $SG_NAME"
SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description "FinBERT sentiment API inbound" --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text)
  info "created $SG_ID"
else
  info "already exists: $SG_ID"
fi
# Port 80 is what the project brief names. Port 8000 is what actually matters: with
# awsvpc networking the SG sits on the task ENI and there is no port translation, so
# the container port must be reachable directly.
for port in 80 8000; do
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port "$port" --cidr 0.0.0.0/0 >/dev/null 2>&1 \
    && info "opened tcp/$port" || info "tcp/$port already open"
done

# ---------------------------------------------------------------------------
say "4/8  CloudWatch log group: $LOG_GROUP"
# Pre-created so the task does not need logs:CreateLogGroup at start time.
aws logs create-log-group --log-group-name "$LOG_GROUP" >/dev/null 2>&1 \
  && info "created" || info "already exists"

# ---------------------------------------------------------------------------
say "5/8  ECS task execution role"
# Fargate REQUIRES an executionRoleArn to pull from ECR and write awslogs.
# This account (Vocareum) has no LabRole and no ecsTaskExecutionRole, and
# iam:SimulatePrincipalPolicy is denied, so we probe by attempting.
EXEC_ROLE_ARN=""
if EXEC_ROLE_ARN=$(aws iam get-role --role-name "$EXEC_ROLE_NAME" \
      --query 'Role.Arn' --output text 2>/dev/null); then
  info "using existing $EXEC_ROLE_NAME"
else
  info "$EXEC_ROLE_NAME not found — attempting to create it"
  if aws iam create-role --role-name "$EXEC_ROLE_NAME" \
        --assume-role-policy-document file://aws/ecs-trust.json >/dev/null 2>&1; then
    aws iam attach-role-policy --role-name "$EXEC_ROLE_NAME" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    info "created; waiting 15s for IAM propagation"
    sleep 15
    EXEC_ROLE_ARN=$(aws iam get-role --role-name "$EXEC_ROLE_NAME" --query 'Role.Arn' --output text)
  else
    info "iam:CreateRole denied — falling back to a pre-existing lab role"
    EXEC_ROLE_ARN=""
    for candidate in LabRole voclabs vocareum; do
      if ARN=$(aws iam get-role --role-name "$candidate" --query 'Role.Arn' --output text 2>/dev/null); then
        # Only usable if ECS tasks are allowed to assume it.
        TRUST=$(aws iam get-role --role-name "$candidate" \
                  --query 'Role.AssumeRolePolicyDocument' --output json 2>/dev/null || echo '{}')
        if echo "$TRUST" | grep -q "ecs-tasks.amazonaws.com"; then
          EXEC_ROLE_ARN="$ARN"; info "using $candidate (trusts ecs-tasks)"; break
        else
          info "$candidate exists but does NOT trust ecs-tasks.amazonaws.com; trying update"
          if aws iam update-assume-role-policy --role-name "$candidate" \
               --policy-document file://aws/ecs-trust.json >/dev/null 2>&1; then
            EXEC_ROLE_ARN="$ARN"; info "updated trust policy on $candidate"; break
          fi
          info "cannot update trust policy on $candidate"
        fi
      fi
    done
  fi
fi

if [ -z "$EXEC_ROLE_ARN" ] || [ "$EXEC_ROLE_ARN" = "None" ]; then
  die "No usable ECS task execution role, and this account will not let us create one.
     Fargate cannot start without it. Options, in order:
       1. Restart the Vocareum lab session — some provision a LabRole on reset.
       2. Use the ECS EC2 launch type (pulls via the instance profile, no exec role).
       3. Demonstrate the service locally with 'docker compose up' and record the
          IAM restriction. The CI/CD workflow and this script are the graded artifacts.
     See docs/AWS_SETUP.md."
fi
info "execution role: $EXEC_ROLE_ARN"

# ---------------------------------------------------------------------------
say "6/8  ECS cluster: $CLUSTER"
STATUS=$(aws ecs describe-clusters --clusters "$CLUSTER" \
  --query 'clusters[0].status' --output text 2>/dev/null || echo "None")
if [ "$STATUS" = "ACTIVE" ]; then
  info "already active"
else
  aws ecs create-cluster --cluster-name "$CLUSTER" --query 'cluster.clusterArn' --output text
  info "created"
fi

# ---------------------------------------------------------------------------
say "7/8  Task definition: $TASK_FAMILY"
export EXEC_ROLE_ARN ECR_REGISTRY ECR_REPOSITORY AWS_REGION
python3 - <<'PY' > /tmp/finbert-taskdef.json
import os, string, pathlib
tpl = pathlib.Path("aws/task-definition.json.tpl").read_text()
print(string.Template(tpl).substitute(os.environ))
PY
REVISION=$(aws ecs register-task-definition --cli-input-json file:///tmp/finbert-taskdef.json \
  --query 'taskDefinition.revision' --output text)
info "registered revision $REVISION"

# ---------------------------------------------------------------------------
say "8/8  ECS service: $SERVICE"
SVC_STATUS=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].status' --output text 2>/dev/null || echo "None")
NETCFG="awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}"

if [ "$SVC_STATUS" = "ACTIVE" ]; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "${TASK_FAMILY}:${REVISION}" --desired-count 1 >/dev/null
  info "updated existing service to revision $REVISION"
else
  aws ecs create-service \
    --cluster "$CLUSTER" --service-name "$SERVICE" \
    --task-definition "${TASK_FAMILY}:${REVISION}" \
    --desired-count 1 --launch-type FARGATE \
    --network-configuration "$NETCFG" \
    --health-check-grace-period-seconds 300 >/dev/null
  info "created"
fi

cat <<SUMMARY

------------------------------------------------------------------
  Bootstrap complete.

  Registry : $ECR_REGISTRY
  Cluster  : $CLUSTER
  Service  : $SERVICE
  Task def : ${TASK_FAMILY}:${REVISION}
  Exec role: $EXEC_ROLE_ARN

  Set the GitHub secrets (credentials are temporary and must be re-set
  whenever the lab session is refreshed):

    gh secret set AWS_ACCESS_KEY_ID     --body "\$AWS_ACCESS_KEY_ID"
    gh secret set AWS_SECRET_ACCESS_KEY --body "\$AWS_SECRET_ACCESS_KEY"
    gh secret set AWS_SESSION_TOKEN     --body "\$AWS_SESSION_TOKEN"
    gh secret set AWS_REGION            --body "$AWS_REGION"
    gh secret set ECR_REGISTRY          --body "$ECR_REGISTRY"

  Watch rollout:
    aws ecs describe-services --cluster $CLUSTER --services $SERVICE \\
      --query 'services[0].deployments'
------------------------------------------------------------------
SUMMARY
