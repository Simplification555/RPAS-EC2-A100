#!/usr/bin/env bash
# A third endpoint on an existing 80GB allocation, activated after external guarding.
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'usage: script <fresh-activation-marker>' >&2; exit 2; }
activation="$1"
[[ ! -e "$activation" ]] || exit 2
cd /home/jianbaizhao/RPAS-EC2-A100
root="$PWD/outputs/scir_ec1_single_reference"
mkdir -p "$root/_jobs" "$root/_locks"
exec 9>"$root/_locks/single_reference.lock"
flock -n 9 || exit 2
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || exit 2
for seed in 0 1 2; do [[ ! -e "$root/single/seed_$seed" ]] || exit 2; done
export RPAS_SCIR_ALLOCATED_GPU=1
export RPAS_EC1_GPU="$CUDA_VISIBLE_DEVICES"
export GEPA_QWEN_DISABLE_THINKING=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
port=$((42000 + SLURM_JOB_ID % 1000))
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:$port/v1"
py() { ./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python "$@"; }
service_pid=''
cleanup() {
  if [[ -n "$service_pid" ]]; then kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
echo "wrapper_pid=$$ port=$port waiting_for_activation=$activation"
for _ in $(seq 1 600); do
  [[ -f "$activation" ]] && break
  sleep 1
done
[[ -f "$activation" ]] || { echo 'No external guard activation; aborting.' >&2; exit 2; }
py -m external_comparison.runners.ec1_preflight \
  --dataset-path data/ec1_humaneval/official/humaneval.jsonl \
  --public-test-path data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  --aflow-validate-path data/ec1_humaneval/aflow/humaneval_validate.jsonl \
  --aflow-test-path data/ec1_humaneval/aflow/humaneval_test.jsonl
./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python scripts/serve_transformers_qwen35_openai.py \
  --model /home/jianbaizhao/model/Qwen/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$port" --max-new-tokens 1024 --max-batch-size 1 \
  --batch-wait-ms 0 --dtype float16 --stop-string '<<RPAS_END>>' \
  >"$root/_jobs/service_${SLURM_JOB_ID}.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 180); do
  curl --max-time 2 -sf "http://127.0.0.1:$port/health" >"$root/_jobs/health.json" && break
  kill -0 "$service_pid" 2>/dev/null || exit 2
  sleep 5
done
curl --max-time 5 -sf "http://127.0.0.1:$port/health" >/dev/null
status=0
for seed in 0 1 2; do
  job_root="$root/_jobs/seed_${seed}_${SLURM_JOB_ID}"
  mkdir -p "$job_root"
  {
    echo "run_started_at_epoch=$(date +%s.%N)"
    echo "job_id=$SLURM_JOB_ID seed=$seed method=single port=$port packed=true"
    nvidia-smi --query-gpu=uuid,name,memory.total --format=csv
    sha256sum external_comparison/runners/single_humaneval.py
  } >"$job_root/environment.txt"
  rc=0
  py -m external_comparison.runners.single_humaneval --seed "$seed" \
    --dataset-path data/ec1_humaneval/official/humaneval.jsonl \
    --public-test-path data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
    --aflow-validate-path data/ec1_humaneval/aflow/humaneval_validate.jsonl \
    --aflow-test-path data/ec1_humaneval/aflow/humaneval_test.jsonl \
    --output-dir "$root" >"$job_root/runner.log" 2>&1 || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    py -m external_comparison.runners.write_run_metrics --result-dir "$root/single/seed_$seed" \
      --job-root "$job_root" --job-id "$SLURM_JOB_ID" --stage formal_single_packed || status=1
    ( cd "$root/single/seed_$seed"
      find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS )
  else
    status=1
  fi
  echo "$(date -Is) single_seed=$seed exit_code=$rc"
done
cleanup
service_pid=''
exit "$status"
