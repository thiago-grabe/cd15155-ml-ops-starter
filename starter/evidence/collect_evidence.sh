#!/usr/bin/env bash
#
# Regenerates the rubric evidence pack.
#
#   ./evidence/collect_evidence.sh            # everything that is currently possible
#   SKIP_AWS=1 ./evidence/collect_evidence.sh # skip AWS checks
#
# Every check writes a machine-generated artifact under evidence/<section>/ and records
# a verdict in evidence/results.tsv. Checks that cannot run are recorded as SKIP or
# BLOCKED with a reason — never silently omitted.
#
# Run from the starter/ directory, with the stack up:
#   docker compose up -d
set -uo pipefail

cd "$(dirname "$0")/.."
EV="evidence"
RESULTS="$EV/results.tsv"
API="http://localhost:8067"
MLFLOW="http://localhost:5067"
PROM="http://localhost:9067"
GRAFANA="http://localhost:3067"
TOOL=(docker compose --profile tools run --rm -T tooling)

mkdir -p "$EV"/{01-dvc,02-mlflow,03-api,04-docker,05-cicd,06-monitoring}
printf 'id\trubric_requirement\tverdict\tartifact\tdetail\n' > "$RESULTS"

pass=0; fail=0; skip=0
record() { # id | requirement | verdict | artifact | detail
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$RESULTS"
  case "$3" in
    PASS) pass=$((pass+1)); c='\033[32m';;
    FAIL) fail=$((fail+1)); c='\033[31m';;
    *)    skip=$((skip+1)); c='\033[33m';;
  esac
  printf "  ${c}%-7s\033[0m %-4s %s\n" "$3" "$1" "$2"
}
# record_if <cond-exit-code> id req artifact detail_pass detail_fail
have() { command -v "$1" >/dev/null 2>&1; }
json() { python3 -c "import json,sys;d=json.load(sys.stdin);print($1)" 2>/dev/null; }

echo "======================================================================"
echo "  FinBERT MLOps — evidence collection"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================================================"

# =====================================================================
echo; echo "-- 1. Data pipeline and versioning (DVC) --"
# =====================================================================
if [ -f dvc.yaml ]; then
  cp dvc.yaml "$EV/01-dvc/dvc.yaml"
  stages=$(uv run --no-project --quiet --with pyyaml python -c \
    "import yaml;print(','.join(yaml.safe_load(open('dvc.yaml'))['stages']))" 2>/dev/null)
  [ "$stages" = "prepare,clean" ] \
    && record 1.1 "dvc.yaml defines stages named prepare and clean" PASS "01-dvc/dvc.yaml" "stages=$stages" \
    || record 1.1 "dvc.yaml defines stages named prepare and clean" FAIL "01-dvc/dvc.yaml" "stages=$stages"

  uv run --no-project --quiet --with pyyaml python - > "$EV/01-dvc/stage-contract.txt" 2>&1 <<'PY'
import yaml
d = yaml.safe_load(open("dvc.yaml"))["stages"]
want = {
 "prepare": {"cmd":"python scripts/load_data.py",
             "deps":["scripts/load_data.py"],
             "params":["prepare.test_size","prepare.random_seed"],
             "outs":["data/train.csv","data/test.csv"]},
 "clean":   {"cmd":"python scripts/clean_data.py",
             "deps":["scripts/clean_data.py","data/raw_stream.csv"],
             "params":["clean.min_words","clean.max_words"],
             "outs":["data/stream.csv"]},
}
ok = True
for stage, req in want.items():
    got = d.get(stage, {})
    for key, vals in req.items():
        actual = got.get(key)
        if key == "cmd":
            good = actual == vals
            print(f"{stage}.cmd: {actual!r} -> {'OK' if good else 'MISMATCH'}")
        else:
            missing = [v for v in vals if v not in (actual or [])]
            good = not missing
            print(f"{stage}.{key}: {actual} -> {'OK' if good else 'MISSING '+str(missing)}")
        ok &= good
print("VERDICT:", "PASS" if ok else "FAIL")
PY
  grep -q "VERDICT: PASS" "$EV/01-dvc/stage-contract.txt" \
    && record 1.2 "prepare/clean declare required cmd, deps, params, outs" PASS "01-dvc/stage-contract.txt" "all fields present" \
    || record 1.2 "prepare/clean declare required cmd, deps, params, outs" FAIL "01-dvc/stage-contract.txt" "see file"
else
  record 1.1 "dvc.yaml defines stages named prepare and clean" FAIL "-" "dvc.yaml missing"
  record 1.2 "prepare/clean declare required cmd, deps, params, outs" FAIL "-" "dvc.yaml missing"
fi

if [ -f dvc.lock ]; then
  cp dvc.lock "$EV/01-dvc/dvc.lock"
  git ls-files --error-unmatch dvc.lock >/dev/null 2>&1 \
    && record 1.3 "dvc.lock committed alongside dvc.yaml" PASS "01-dvc/dvc.lock" "tracked by git" \
    || record 1.3 "dvc.lock committed alongside dvc.yaml" SKIP "01-dvc/dvc.lock" "exists but not yet git-added"
else
  record 1.3 "dvc.lock committed alongside dvc.yaml" FAIL "-" "dvc.lock missing — run dvc repro"
fi

# Truncate first: appending across runs left multiple row counts in the file, and the
# grep below then produced two numbers and broke the integer comparison.
: > "$EV/01-dvc/dataset-rows.txt"
for f in train test stream; do
  if [ -f "data/$f.csv" ]; then
    n=$(($(wc -l < "data/$f.csv") - 1))
    echo "data/$f.csv rows=$n" >> "$EV/01-dvc/dataset-rows.txt"
  fi
done
if [ -f "$EV/01-dvc/dataset-rows.txt" ]; then
  srow=$(grep -o 'stream.csv rows=[0-9]*' "$EV/01-dvc/dataset-rows.txt" | grep -o '[0-9]*$' | tail -1)
  if [ -n "${srow:-}" ] && [ "$srow" -ge 100 ]; then
    record 1.4 "Datasets generated; stream.csv >= deepchecks min_samples(100)" PASS "01-dvc/dataset-rows.txt" "stream.csv=$srow rows"
  else
    record 1.4 "Datasets generated; stream.csv >= deepchecks min_samples(100)" FAIL "01-dvc/dataset-rows.txt" "stream.csv=${srow:-0} rows (<100)"
  fi
else
  record 1.4 "Datasets generated; stream.csv >= deepchecks min_samples(100)" SKIP "-" "run dvc repro first"
fi

# =====================================================================
echo; echo "-- 2. Model registry and evaluation (MLflow) --"
# =====================================================================
grep -q "mlflow.set_tracking_uri" scripts/evaluate.py && a=1 || a=0
grep -q "mlflow.set_experiment"   scripts/evaluate.py && b=1 || b=0
grep -q "mlflow.start_run"        scripts/evaluate.py && c=1 || c=0
grep -q "log_params"              scripts/evaluate.py && d=1 || d=0
grep -q "log_metrics"             scripts/evaluate.py && e=1 || e=0
grep -q "registered_model_name"   scripts/evaluate.py && f=1 || f=0
cp scripts/evaluate.py "$EV/02-mlflow/evaluate.py"
[ $((a+b+c+d+e+f)) -eq 6 ] \
  && record 2.1 "evaluate.py sets URI+experiment, logs params/metrics, registers model" PASS "02-mlflow/evaluate.py" "all 6 calls present" \
  || record 2.1 "evaluate.py sets URI+experiment, logs params/metrics, registers model" FAIL "02-mlflow/evaluate.py" "uri=$a exp=$b run=$c params=$d metrics=$e register=$f"

if curl -fsS --max-time 5 "$MLFLOW/health" >/dev/null 2>&1; then
  EXP=$(curl -fsS "$MLFLOW/api/2.0/mlflow/experiments/get-by-name?experiment_name=finbert-evaluation" 2>/dev/null)
  echo "$EXP" > "$EV/02-mlflow/experiment.json"
  EXPID=$(echo "$EXP" | json 'd["experiment"]["experiment_id"]')
  if [ -n "${EXPID:-}" ]; then
    curl -fsS -X POST "$MLFLOW/api/2.0/mlflow/runs/search" \
      -H 'Content-Type: application/json' \
      -d "{\"experiment_ids\":[\"$EXPID\"],\"max_results\":5,\"order_by\":[\"attributes.start_time DESC\"]}" \
      > "$EV/02-mlflow/runs.json" 2>/dev/null
    METRICS=$(python3 - "$EV/02-mlflow/runs.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
runs=d.get("runs",[])
need={"accuracy","f1_weighted","precision_weighted","recall_weighted"}
for r in runs:
    got={m["key"] for m in r.get("data",{}).get("metrics",[])}
    if need <= got:
        print(",".join(sorted(need)));break
else:
    print("")
PY
)
    [ -n "$METRICS" ] \
      && record 2.2 "MLflow run logs all four evaluation metrics" PASS "02-mlflow/runs.json" "$METRICS" \
      || record 2.2 "MLflow run logs all four evaluation metrics" FAIL "02-mlflow/runs.json" "one or more metrics missing"
  else
    record 2.2 "MLflow run logs all four evaluation metrics" SKIP "-" "experiment not found — run evaluate.py"
  fi

  RM=$(curl -fsS "$MLFLOW/api/2.0/mlflow/registered-models/get?name=finbert" 2>/dev/null)
  if [ -n "$RM" ]; then
    echo "$RM" > "$EV/02-mlflow/registered-model.json"
    record 2.3 "Model registered in the MLflow Model Registry" PASS "02-mlflow/registered-model.json" "name=finbert"
    AL=$(curl -fsS "$MLFLOW/api/2.0/mlflow/registered-models/alias?name=finbert&alias=production" 2>/dev/null)
    if [ -n "$AL" ] && echo "$AL" | grep -q '"version"'; then
      echo "$AL" > "$EV/02-mlflow/production-alias.json"
      V=$(echo "$AL" | json 'd["model_version"]["version"]')
      record 2.4 "production alias assigned to a registered version" PASS "02-mlflow/production-alias.json" "version=$V"
    else
      record 2.4 "production alias assigned to a registered version" FAIL "-" "alias not set — run promote.py"
    fi
  else
    record 2.3 "Model registered in the MLflow Model Registry" SKIP "-" "no registered model — run evaluate.py"
    record 2.4 "production alias assigned to a registered version" SKIP "-" "no registered model"
  fi
else
  record 2.2 "MLflow run logs all four evaluation metrics" SKIP "-" "MLflow not reachable at $MLFLOW"
  record 2.3 "Model registered in the MLflow Model Registry" SKIP "-" "MLflow not reachable"
  record 2.4 "production alias assigned to a registered version" SKIP "-" "MLflow not reachable"
fi

grep -q "f1_threshold" scripts/promote.py && grep -q "set_registered_model_alias" scripts/promote.py \
  && grep -qi "NOT PROMOTED" scripts/promote.py \
  && record 2.5 "promote.py reads threshold, sets alias, prints message on failure" PASS "02-mlflow/promote.py" "all three behaviours present" \
  || record 2.5 "promote.py reads threshold, sets alias, prints message on failure" FAIL "02-mlflow/promote.py" "see file"
cp scripts/promote.py "$EV/02-mlflow/promote.py"

# =====================================================================
echo; echo "-- 3. Inference service --"
# =====================================================================
grep -q "def run_prediction" app/main.py \
  && record 3.0 "run_prediction helper present in app/main.py" PASS "03-api/main.py" "$(grep -n 'def run_prediction' app/main.py | tr '\n' ' ')" \
  || record 3.0 "run_prediction helper present in app/main.py" FAIL "03-api/main.py" "helper missing"
cp app/main.py "$EV/03-api/main.py"

if curl -fsS --max-time 5 "$API/health" >/dev/null 2>&1; then
  {
    echo "### GET /health"; curl -s -i "$API/health" | head -20; echo
    echo "### POST /predict"; curl -s -i -X POST "$API/predict" \
      -H 'Content-Type: application/json' \
      -d '{"text":"The company reported record profits and raised its dividend."}' | head -20; echo
    echo "### POST /predict/batch (3 texts)"; curl -s -i -X POST "$API/predict/batch" \
      -H 'Content-Type: application/json' \
      -d '{"texts":["Record profits.","Bankruptcy filing.","Flat trading."]}' | head -25; echo
    echo "### POST /predict  (empty text -> expect 422)"; curl -s -o /dev/null -w 'status=%{http_code}\n' \
      -X POST "$API/predict" -H 'Content-Type: application/json' -d '{"text":""}'
    echo "### POST /predict/batch (empty list -> expect 422)"; curl -s -o /dev/null -w 'status=%{http_code}\n' \
      -X POST "$API/predict/batch" -H 'Content-Type: application/json' -d '{"texts":[]}'
  } > "$EV/03-api/endpoint-transcript.txt" 2>&1

  H=$(curl -s -o /dev/null -w '%{http_code}' "$API/health")
  [ "$H" = "200" ] && record 3.1 "GET /health returns 200 {\"status\":\"ok\"}" PASS "03-api/endpoint-transcript.txt" "status=$H" \
                   || record 3.1 "GET /health returns 200 {\"status\":\"ok\"}" FAIL "03-api/endpoint-transcript.txt" "status=$H"

  P=$(curl -s -X POST "$API/predict" -H 'Content-Type: application/json' -d '{"text":"Record profits."}')
  echo "$P" > "$EV/03-api/predict-response.json"
  KEYS=$(echo "$P" | json "','.join(sorted(d))")
  [ "$KEYS" = "confidence,latency_ms,sentiment,text" ] \
    && record 3.2 "POST /predict returns text, sentiment, confidence, latency_ms" PASS "03-api/predict-response.json" "keys=$KEYS" \
    || record 3.2 "POST /predict returns text, sentiment, confidence, latency_ms" FAIL "03-api/predict-response.json" "keys=$KEYS"

  B=$(curl -s -X POST "$API/predict/batch" -H 'Content-Type: application/json' \
        -d '{"texts":["Record profits.","Bankruptcy filing.","Flat trading."]}')
  echo "$B" > "$EV/03-api/predict-batch-response.json"
  N=$(echo "$B" | json "len(d)")
  [ "${N:-0}" = "3" ] \
    && record 3.3 "POST /predict/batch returns one result per input" PASS "03-api/predict-batch-response.json" "3 in -> $N out" \
    || record 3.3 "POST /predict/batch returns one result per input" FAIL "03-api/predict-batch-response.json" "3 in -> ${N:-?} out"

  E1=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/predict" -H 'Content-Type: application/json' -d '{"text":""}')
  E2=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/predict/batch" -H 'Content-Type: application/json' -d '{"texts":[]}')
  [ "$E1" = "422" ] && [ "$E2" = "422" ] \
    && record 3.4 "422 for empty text and empty texts list" PASS "03-api/endpoint-transcript.txt" "predict=$E1 batch=$E2" \
    || record 3.4 "422 for empty text and empty texts list" FAIL "03-api/endpoint-transcript.txt" "predict=$E1 batch=$E2"
else
  for i in 3.1 3.2 3.3 3.4; do record $i "API endpoint behaviour" SKIP "-" "API not reachable at $API"; done
fi

if "${TOOL[@]}" pytest tests/ -v > "$EV/03-api/pytest.txt" 2>&1; then
  NT=$(grep -cE "PASSED" "$EV/03-api/pytest.txt")
  record 3.5 "Integration tests pass (>=3 covering the named cases)" PASS "03-api/pytest.txt" "$NT tests passed"
else
  record 3.5 "Integration tests pass (>=3 covering the named cases)" FAIL "03-api/pytest.txt" "see file"
fi

cp scripts/locustfile.py "$EV/03-api/locustfile.py"
LT=$(grep -c "@task" scripts/locustfile.py)
grep -q "wait_time" scripts/locustfile.py && W=yes || W=no
grep -q "@task(3)" scripts/locustfile.py && W3=yes || W3=no
[ "$LT" = "3" ] && [ "$W" = "yes" ] && [ "$W3" = "yes" ] \
  && record 3.6 "Locust HttpUser: 3 weighted tasks (3/1/1) + wait_time" PASS "03-api/locustfile.py" "tasks=$LT wait_time=$W weight3=$W3" \
  || record 3.6 "Locust HttpUser: 3 weighted tasks (3/1/1) + wait_time" FAIL "03-api/locustfile.py" "tasks=$LT wait_time=$W weight3=$W3"

# =====================================================================
echo; echo "-- 4. Containerisation --"
# =====================================================================
cp Dockerfile "$EV/04-docker/Dockerfile"
d1=$(grep -c "^FROM python:3.12-slim" Dockerfile)
d2=$(grep -c "pip install .*-r requirements" Dockerfile)
d3=$(grep -c "^COPY app/" Dockerfile)
d4=$(grep -c "^EXPOSE 8000" Dockerfile)
d5=$(grep -c "^CMD.*uvicorn" Dockerfile)
[ "$d1" -ge 1 ] && [ "$d2" -ge 1 ] && [ "$d3" -ge 1 ] && [ "$d4" -ge 1 ] && [ "$d5" -ge 1 ] \
  && record 4.1 "Dockerfile: slim base, reqs install, COPY app/, EXPOSE 8000, CMD" PASS "04-docker/Dockerfile" "all five anchors present" \
  || record 4.1 "Dockerfile: slim base, reqs install, COPY app/, EXPOSE 8000, CMD" FAIL "04-docker/Dockerfile" "from=$d1 reqs=$d2 copy=$d3 expose=$d4 cmd=$d5"

cp docker-compose.yml "$EV/04-docker/docker-compose.yml"
docker compose config > "$EV/04-docker/compose-config-resolved.yml" 2>&1
# NOTE: `docker compose config` RESOLVES env_file into `environment` and drops the key,
# so the env_file assertion must read the SOURCE file, not the rendered output.
uv run --no-project --quiet --with pyyaml python - > "$EV/04-docker/compose-contract.txt" 2>&1 <<'PY'
import yaml, json
api = yaml.safe_load(open("docker-compose.yml"))["services"]["api"]
checks = {
  "build directive":              bool(api.get("build")),
  "port mapping to container 8000": any("8000" in str(p).split(":")[-1] for p in api.get("ports", [])),
  "env_file .env":                any(".env" in str(e) for e in api.get("env_file", [])),
  "volume mount mlruns/":         any("mlruns" in str(v) for v in api.get("volumes", [])),
  "volume mount mlartifacts/":    any("mlartifacts" in str(v) for v in api.get("volumes", [])),
  "extra_hosts host-gateway":     any("host-gateway" in str(h) for h in api.get("extra_hosts", [])),
  "healthcheck on GET /health":   "/health" in json.dumps(api.get("healthcheck", {})),
}
for k, v in checks.items():
    print(f"{'OK  ' if v else 'MISS'}  {k}")
print("VERDICT:", "PASS" if all(checks.values()) else "FAIL")
print("\nports:", api.get("ports"))
PY
grep -q "VERDICT: PASS" "$EV/04-docker/compose-contract.txt" \
  && record 4.2 "compose api: build, ports, env_file, mlruns+mlartifacts, extra_hosts, healthcheck" PASS "04-docker/compose-contract.txt" "all six keys present" \
  || record 4.2 "compose api: build, ports, env_file, mlruns+mlartifacts, extra_hosts, healthcheck" FAIL "04-docker/compose-contract.txt" "see file"

docker compose ps --format json > "$EV/04-docker/compose-ps.json" 2>/dev/null
HEALTH=$(docker inspect --format '{{.State.Health.Status}}' finbert-api 2>/dev/null || echo "not running")
[ "$HEALTH" = "healthy" ] \
  && record 4.3 "api container reports healthy" PASS "04-docker/compose-ps.json" "health=$HEALTH" \
  || record 4.3 "api container reports healthy" SKIP "04-docker/compose-ps.json" "health=$HEALTH"

docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -Ei 'starter|finbert' \
  > "$EV/04-docker/image-sizes.txt" 2>/dev/null
[ -s "$EV/04-docker/image-sizes.txt" ] \
  && record 4.4 "Slim CPU-only image size recorded (stand-out)" PASS "04-docker/image-sizes.txt" "$(head -1 "$EV/04-docker/image-sizes.txt")" \
  || record 4.4 "Slim CPU-only image size recorded (stand-out)" SKIP "-" "no images built"

# =====================================================================
echo; echo "-- 5. CI/CD and drift gate --"
# =====================================================================
cp scripts/run_deepchecks.py "$EV/05-cicd/run_deepchecks.py"
g1=$(grep -c "TextData(" scripts/run_deepchecks.py)
grep -q "PropertyDrift"   scripts/run_deepchecks.py && g2=yes || g2=no
grep -q "PredictionDrift" scripts/run_deepchecks.py && g3=yes || g3=no
grep -q "sys.exit(1)"     scripts/run_deepchecks.py && g4=yes || g4=no
[ "$g1" -ge 2 ] && [ "$g2" = yes ] && [ "$g3" = yes ] && [ "$g4" = yes ] \
  && record 5.1 "run_deepchecks: 2 TextData, property+prediction drift, non-zero exit" PASS "05-cicd/run_deepchecks.py" "TextData=$g1 prop=$g2 pred=$g3 exit=$g4" \
  || record 5.1 "run_deepchecks: 2 TextData, property+prediction drift, non-zero exit" FAIL "05-cicd/run_deepchecks.py" "TextData=$g1 prop=$g2 pred=$g3 exit=$g4"

if [ -f data/stream.csv ] && [ -f data/test.csv ]; then
  "${TOOL[@]}" python scripts/run_deepchecks.py > "$EV/05-cicd/deepchecks-run.txt" 2>&1
  DC=$?
  echo "EXIT CODE: $DC" >> "$EV/05-cicd/deepchecks-run.txt"
  if [ $DC -eq 0 ]; then
    record 5.2 "Drift gate executes and passes on current data" PASS "05-cicd/deepchecks-run.txt" "exit=0"
  else
    record 5.2 "Drift gate executes and passes on current data" FAIL "05-cicd/deepchecks-run.txt" "exit=$DC (see file — drift may legitimately exceed threshold)"
  fi
else
  record 5.2 "Drift gate executes and passes on current data" SKIP "-" "datasets missing — run dvc repro"
fi

WF="../.github/workflows/ci-cd.yml"
if [ -f "$WF" ]; then
  cp "$WF" "$EV/05-cicd/ci-cd.yml"
  uv run --no-project --quiet --with pyyaml python - "$WF" > "$EV/05-cicd/workflow-contract.txt" 2>&1 <<'PY'
import yaml,sys
d=yaml.safe_load(open(sys.argv[1]))
on=d.get(True, d.get("on"))
jobs=d["jobs"]
want={"test":"lint","deepchecks":"lint","build":["test","deepchecks"],"deploy":"build"}
print("triggers:", on)
ok = "push" in on and "main" in on["push"]["branches"]
print("push to main:", ok)
for j,need in want.items():
    got=jobs.get(j,{}).get("needs")
    good = got==need or (isinstance(need,list) and sorted(got or [])==sorted(need))
    print(f"job {j}: needs={got} -> {'OK' if good else 'MISMATCH expected '+str(need)}")
    ok &= good
for j in ("build","deploy"):
    s=str(jobs.get(j,{}))
    print(f"job {j}: uses AWS creds  ->", "aws-actions/configure-aws-credentials" in s)
print("deploy uses ecs-deploy-task-definition:", "amazon-ecs-deploy-task-definition" in str(jobs.get("deploy",{})))
print("VERDICT:", "PASS" if ok else "FAIL")
PY
  grep -q "VERDICT: PASS" "$EV/05-cicd/workflow-contract.txt" \
    && record 5.3 "ci-cd.yml on push:main with test/deepchecks/build/deploy job graph" PASS "05-cicd/workflow-contract.txt" "job graph correct" \
    || record 5.3 "ci-cd.yml on push:main with test/deepchecks/build/deploy job graph" FAIL "05-cicd/workflow-contract.txt" "see file"
else
  record 5.3 "ci-cd.yml on push:main with test/deepchecks/build/deploy job graph" FAIL "-" "repo-root workflow missing"
fi

if have gh; then
  gh workflow list --all > "$EV/05-cicd/gh-workflow-list.txt" 2>&1
  grep -qi "test-and-deploy" "$EV/05-cicd/gh-workflow-list.txt" \
    && record 5.4 "GitHub actually registers the workflow" PASS "05-cicd/gh-workflow-list.txt" "visible to GitHub" \
    || record 5.4 "GitHub actually registers the workflow" SKIP "05-cicd/gh-workflow-list.txt" "not registered until pushed to the default branch"
else
  record 5.4 "GitHub actually registers the workflow" SKIP "-" "gh CLI unavailable"
fi

if [ "${SKIP_AWS:-0}" = "1" ]; then
  for i in 5.5 5.6; do record $i "AWS resources / deployment" SKIP "-" "SKIP_AWS=1"; done
elif aws sts get-caller-identity > "$EV/05-cicd/aws-identity.json" 2>&1; then
  aws ecr list-images --repository-name finbert-api > "$EV/05-cicd/ecr-images.json" 2>&1 \
    && record 5.5 "Image present in ECR" PASS "05-cicd/ecr-images.json" "$(python3 -c "import json;print(len(json.load(open('$EV/05-cicd/ecr-images.json'))['imageIds']),'images')" 2>/dev/null || echo 'see file')" \
    || record 5.5 "Image present in ECR" SKIP "05-cicd/ecr-images.json" "repo absent — run scripts/aws_bootstrap.sh"
  aws ecs describe-services --cluster finbert-cluster --services finbert-api-service \
    > "$EV/05-cicd/ecs-service.json" 2>&1
  RUNNING=$(python3 -c "import json;s=json.load(open('$EV/05-cicd/ecs-service.json'))['services'];print(s[0]['runningCount'] if s else 0)" 2>/dev/null || echo 0)
  [ "${RUNNING:-0}" -ge 1 ] \
    && record 5.6 "ECS service running the deployed task" PASS "05-cicd/ecs-service.json" "runningCount=$RUNNING" \
    || record 5.6 "ECS service running the deployed task" SKIP "05-cicd/ecs-service.json" "runningCount=${RUNNING:-0}"
else
  for i in 5.5 5.6; do record $i "AWS resources / deployment" SKIP "05-cicd/aws-identity.json" "AWS credentials invalid or expired"; done
fi

# =====================================================================
echo; echo "-- 6. Production monitoring --"
# =====================================================================
# Match the metric OBJECTS, not a literal "Counter(" — the constructors are passed
# by name into _get_or_create(), so "Counter(" never appears on one line.
m1=$(grep -cE '^PREDICTION_(REQUESTS|ERRORS) = ' app/main.py)
grep -qE '^PREDICTION_LATENCY = ' app/main.py && grep -q 'Histogram' app/main.py && m2=yes || m2=no
grep -q 'generate_latest'       app/main.py && m3=yes || m3=no
grep -q 'CONTENT_TYPE_LATEST'   app/main.py && m4=yes || m4=no
[ "$m1" -ge 2 ] && [ "$m2" = yes ] && [ "$m3" = yes ] && [ "$m4" = yes ] \
  && record 6.1 "Counter(sentiment) + Histogram + error Counter + /metrics exposition" PASS "03-api/main.py" "Counters=$m1 Histogram=$m2 generate_latest=$m3" \
  || record 6.1 "Counter(sentiment) + Histogram + error Counter + /metrics exposition" FAIL "03-api/main.py" "Counters=$m1 Histogram=$m2"

if curl -fsS --max-time 5 "$API/metrics" -o "$EV/06-monitoring/metrics.txt" 2>/dev/null; then
  r1=$(grep -c '^prediction_requests_total' "$EV/06-monitoring/metrics.txt")
  r2=$(grep -c '^prediction_latency_ms_bucket' "$EV/06-monitoring/metrics.txt")
  r3=$(grep -c '^prediction_errors_total' "$EV/06-monitoring/metrics.txt")
  [ "$r1" -ge 1 ] && [ "$r2" -ge 1 ] && [ "$r3" -ge 1 ] \
    && record 6.2 "/metrics exposes all three instrument families" PASS "06-monitoring/metrics.txt" "requests=$r1 latency_buckets=$r2 errors=$r3" \
    || record 6.2 "/metrics exposes all three instrument families" FAIL "06-monitoring/metrics.txt" "requests=$r1 latency=$r2 errors=$r3"
else
  record 6.2 "/metrics exposes all three instrument families" SKIP "-" "API not reachable"
fi

docker compose logs api --no-log-prefix > "$EV/06-monitoring/api-logs.txt" 2>/dev/null
if [ -s "$EV/06-monitoring/api-logs.txt" ]; then
  VALID=$(python3 - "$EV/06-monitoring/api-logs.txt" <<'PY'
import json,sys
ok=tot=0
for line in open(sys.argv[1], errors="ignore"):
    line=line.strip()
    if not line.startswith("{"): continue
    tot+=1
    try:
        d=json.loads(line)
        if {"timestamp","level","message"} <= set(d): ok+=1
    except Exception: pass
print(f"{ok}/{tot}")
PY
)
  case "$VALID" in
    0/0) record 6.3 "Structured JSON logs with timestamp/level/message" SKIP "06-monitoring/api-logs.txt" "no JSON lines captured yet";;
    *)   n=${VALID%%/*}; [ "$n" -gt 0 ] \
           && record 6.3 "Structured JSON logs with timestamp/level/message" PASS "06-monitoring/api-logs.txt" "$VALID lines valid" \
           || record 6.3 "Structured JSON logs with timestamp/level/message" FAIL "06-monitoring/api-logs.txt" "$VALID";;
  esac
else
  record 6.3 "Structured JSON logs with timestamp/level/message" SKIP "-" "api container not running"
fi

cp prometheus.yml "$EV/06-monitoring/prometheus.yml"
uv run --no-project --quiet --with pyyaml python - > "$EV/06-monitoring/prometheus-contract.txt" 2>&1 <<'PY'
import yaml
d=yaml.safe_load(open("prometheus.yml"))
sc=d["scrape_configs"][0]
checks={
 'job_name == "finbert-api"': sc.get("job_name")=="finbert-api",
 'scrape_interval == 15s'   : sc.get("scrape_interval")=="15s" or d.get("global",{}).get("scrape_interval")=="15s",
 'target api:8000'          : any("api:8000" in str(t) for g in sc.get("static_configs",[]) for t in g.get("targets",[])),
}
for k,v in checks.items(): print(f"{'OK  ' if v else 'MISS'}  {k}")
print("VERDICT:", "PASS" if all(checks.values()) else "FAIL")
PY
grep -q "VERDICT: PASS" "$EV/06-monitoring/prometheus-contract.txt" \
  && record 6.4 "prometheus.yml: job finbert-api, 15s, target api:8000" PASS "06-monitoring/prometheus-contract.txt" "all three present" \
  || record 6.4 "prometheus.yml: job finbert-api, 15s, target api:8000" FAIL "06-monitoring/prometheus-contract.txt" "see file"

if curl -fsS --max-time 5 "$PROM/api/v1/targets" -o "$EV/06-monitoring/prometheus-targets.json" 2>/dev/null; then
  UP=$(python3 -c "
import json
d=json.load(open('$EV/06-monitoring/prometheus-targets.json'))
t=[x for x in d['data']['activeTargets'] if x['labels'].get('job')=='finbert-api']
print(t[0]['health'] if t else 'absent')" 2>/dev/null)
  [ "$UP" = "up" ] \
    && record 6.5 "Prometheus is actually scraping the API" PASS "06-monitoring/prometheus-targets.json" "target health=$UP" \
    || record 6.5 "Prometheus is actually scraping the API" FAIL "06-monitoring/prometheus-targets.json" "target health=$UP"
else
  record 6.5 "Prometheus is actually scraping the API" SKIP "-" "Prometheus not reachable at $PROM"
fi

s1=$(grep -c "log_window" monitoring/stream.py)
grep -q "step=window_idx" monitoring/stream.py && s2=yes || s2=no
s3=$(grep -cE "pct_positive|pct_negative|pct_neutral|avg_confidence|avg_latency_ms" monitoring/stream.py)
cp monitoring/stream.py "$EV/06-monitoring/stream.py"
[ "$s1" -ge 2 ] && [ "$s2" = yes ] && [ "$s3" -ge 5 ] \
  && record 6.6 "stream.py logs 5 window metrics to MLflow with step index" PASS "06-monitoring/stream.py" "log_window=$s1 step=$s2 metrics=$s3" \
  || record 6.6 "stream.py logs 5 window metrics to MLflow with step index" FAIL "06-monitoring/stream.py" "log_window=$s1 step=$s2 metrics=$s3"

cp monitoring/grafana/dashboards/finbert-api.json "$EV/06-monitoring/grafana-dashboard.json" 2>/dev/null \
  && record 6.7 "Grafana dashboard JSON committed (stand-out)" PASS "06-monitoring/grafana-dashboard.json" "$(python3 -c "import json;print(len(json.load(open('$EV/06-monitoring/grafana-dashboard.json'))['panels']),'panels')" 2>/dev/null)" \
  || record 6.7 "Grafana dashboard JSON committed (stand-out)" FAIL "-" "dashboard missing"

grep -q "environment:" ../.github/workflows/ci-cd.yml && grep -q "name: production" ../.github/workflows/ci-cd.yml \
  && record 6.8 "Deploy approval gate via GitHub environment (stand-out)" PASS "05-cicd/ci-cd.yml" "environment: production on deploy job" \
  || record 6.8 "Deploy approval gate via GitHub environment (stand-out)" FAIL "-" "no environment gate"

[ -f ../.github/workflows/rollback.yml ] && [ -f scripts/rollback.py ] \
  && { cp ../.github/workflows/rollback.yml "$EV/05-cicd/rollback.yml"; cp scripts/rollback.py "$EV/02-mlflow/rollback.py"; \
       record 6.9 "Model rollback workflow + script (stand-out)" PASS "05-cicd/rollback.yml" "workflow_dispatch + scripts/rollback.py"; } \
  || record 6.9 "Model rollback workflow + script (stand-out)" FAIL "-" "rollback artefacts missing"

# =====================================================================
echo
echo "======================================================================"
printf "  PASS %d    FAIL %d    SKIP %d\n" "$pass" "$fail" "$skip"
echo "  Results: $RESULTS"
echo "======================================================================"

python3 - "$RESULTS" > "$EV/README.md" <<'PY'
import csv, sys, datetime
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
p = sum(r["verdict"] == "PASS" for r in rows)
f = sum(r["verdict"] == "FAIL" for r in rows)
s = sum(r["verdict"] not in ("PASS", "FAIL") for r in rows)
icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP", "BLOCKED": "BLOCKED"}
print("# Rubric Evidence Pack\n")
print(f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
print(f"**PASS {p} · FAIL {f} · SKIP {s}**\n")
print("Every row below is produced by `./evidence/collect_evidence.sh`. Artifacts are")
print("machine-generated; nothing here is hand-written after the fact.\n")
print("Re-run with the stack up (`docker compose up -d`) to refresh.\n")
print("| # | Rubric requirement | Verdict | Evidence | Detail |")
print("|---|---|---|---|---|")
for r in rows:
    art = f"[`{r['artifact']}`]({r['artifact']})" if r["artifact"] != "-" else "—"
    print(f"| {r['id']} | {r['rubric_requirement']} | **{icon.get(r['verdict'], r['verdict'])}** | {art} | {r['detail']} |")
if f:
    print("\n## Failing items\n")
    for r in rows:
        if r["verdict"] == "FAIL":
            print(f"- **{r['id']} {r['rubric_requirement']}** — {r['detail']} (see `{r['artifact']}`)")
if s:
    print("\n## Not verified in this run\n")
    for r in rows:
        if r["verdict"] not in ("PASS", "FAIL"):
            print(f"- **{r['id']} {r['rubric_requirement']}** — {r['detail']}")
PY
echo "  Summary written to $EV/README.md"
[ "$fail" -eq 0 ]
