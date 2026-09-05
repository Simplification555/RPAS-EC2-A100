#!/usr/bin/env bash
# Three pending seeds, one new model copy; the existing G-Designer service is borrowed.
set -euo pipefail
[[ $# -eq 5 ]] || { echo 'usage: script <parent> <shared-service> <shared-port> <new-port> <activation>' >&2; exit 2; }
parent="$1"; shared_service="$2"; shared_port="$3"; port="$4"; activation="$5"
for value in "$parent" "$shared_service" "$shared_port" "$port"; do [[ "$value" =~ ^[0-9]+$ ]] || exit 2; done
[[ -n "${SLURM_JOB_ID:-}" && "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ && ! -e "$activation" ]] || exit 2
[[ "$shared_port" != "$port" ]] || exit 2
cd /home/jianbaizhao/RPAS-EC2-A100
root="$PWD/outputs/scir_ec2_formal"
for pid in "$parent" "$shared_service"; do
  tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "SLURM_JOB_ID=$SLURM_JOB_ID" >/dev/null
done
tr '\0' ' ' <"/proc/$parent/cmdline" | grep -F '/slurm_script' >/dev/null
tr '\0' ' ' <"/proc/$shared_service/cmdline" | grep -F 'serve_transformers_qwen35_openai.py' >/dev/null
tr '\0' ' ' <"/proc/$shared_service/cmdline" | grep -F -- "--port $shared_port " >/dev/null
parent_start="$(awk '{print $22}' "/proc/$parent/stat")"
service_start="$(awk '{print $22}' "/proc/$shared_service/stat")"
mkdir -p "$root/_locks" "$root/_jobs"
exec 7>"$root/_locks/gdesigner_seed_2.lock"
exec 8>"$root/_locks/rpas_comm_seed_1.lock"
exec 9>"$root/_locks/rpas_comm_seed_2.lock"
flock -n 7 && flock -n 8 && flock -n 9 || exit 2
for relative in gdesigner/seed_2 rpas_comm/seed_1 rpas_comm/seed_2; do
  [[ ! -e "$root/$relative" ]] || { echo "Refusing existing destination $relative" >&2; exit 2; }
done
echo "bundle_wrapper_pid=$$ waiting_for_activation=$activation"
for _ in $(seq 1 600); do
  [[ -f "$activation" ]] && break
  kill -0 "$shared_service" 2>/dev/null || exit 2
  sleep 1
done
[[ -f "$activation" ]] || exit 2
[[ "$(awk '{print $3, $22}' "/proc/$parent/stat")" == "T $parent_start" ]] || exit 2
[[ "$(awk '{print $22}' "/proc/$shared_service/stat")" == "$service_start" ]] || exit 2
guard="$(<"$activation")"
[[ "$guard" =~ ^[0-9]+$ ]] || exit 2
tr '\0' '\n' <"/proc/$guard/environ" | grep -Fx "SLURM_JOB_ID=$SLURM_JOB_ID" >/dev/null
tr '\0' ' ' <"/proc/$guard/cmdline" | grep -F 'guard_packed_allocation.sh' >/dev/null
gpu_free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
[[ "$gpu_free" =~ ^[0-9]+$ && "$gpu_free" -ge 28000 ]] || { echo 'Insufficient GPU headroom' >&2; exit 2; }
curl --max-time 5 -sf "http://127.0.0.1:$shared_port/v1/models" >/dev/null
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RPAS_SCIR_ALLOCATED_GPU=1 RPAS_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export RPAS_MMLU_MAX_TOKENS=256 RPAS_EC2_FIXED_EVAL_CONCURRENCY=4
export RPAS_EXTERNAL_API_KEY=EMPTY RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_TOKENIZER_PATH=/home/jianbaizhao/model/Qwen/Qwen3.5-9B
export RPAS_GDESIGNER_EMBEDDING_MODEL=/home/jianbaizhao/model/sentence-transformers/all-MiniLM-L6-v2
export RPAS_MODEL_CONFIG="$PWD/experiments/phase2_mmlu_qwen35_9b.json"
export GEPA_QWEN_DISABLE_THINKING=1 GEPA_PHASE2_PROMPT_MODE=deliberate
py() { ./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python "$@"; }
service_pid=''
workers=()
cleanup() {
  # Only terminate children owned by this bundle, never the borrowed service.
  for pid in "${workers[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${workers[@]}"; do wait "$pid" 2>/dev/null || true; done
  if [[ -n "$service_pid" ]]; then
    kill -TERM "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python \
  scripts/serve_transformers_qwen35_openai.py --model "$RPAS_TOKENIZER_PATH" \
  --served-model-name Qwen/Qwen3.5-9B --host 127.0.0.1 --port "$port" \
  --max-new-tokens 768 --max-batch-size 4 --batch-wait-ms 25 --dtype float16 --stop-string '<<RPAS_END>>' \
  >"$root/_jobs/pending_bundle_service_$SLURM_JOB_ID.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 180); do
  curl --max-time 2 -sf "http://127.0.0.1:$port/health" >/dev/null && break
  kill -0 "$service_pid" 2>/dev/null || exit 2
  sleep 5
done
curl --max-time 5 -sf "http://127.0.0.1:$port/health" >/dev/null
py scripts/scir/ec2_batch_service_preflight.py --base-url "http://127.0.0.1:$port/v1" \
  --output "$root/_jobs/pending_bundle_batch_preflight_$SLURM_JOB_ID.json"

run_seed() (
  method="$1"; seed="$2"; endpoint_port="$3"
  result_dir="$root/$method/seed_$seed"
  mkdir "$result_dir"
  mkdir "$result_dir/logs"
  export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:$endpoint_port/v1"
  child=''
  stop_child() {
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
  }
  trap stop_child EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  {
    echo "run_started_at_epoch=$(date +%s.%N)"
    echo "job_id=$SLURM_JOB_ID method=$method seed=$seed shared_endpoint=true port=$endpoint_port"
    echo 'timing_scope=co_resident_operational_not_isolated'
    echo "guard_pid=$guard bundle_pid=$$ owner_parent=$parent"
    hostname
    nvidia-smi --query-gpu=uuid,name,memory.total --format=csv
    sha256sum external_comparison/runners/ec2_v2.py scripts/scir/ec2_pending_bundle_packed.sh
  } >"$result_dir/environment.txt"
  curl --max-time 5 -sf "$RPAS_EXTERNAL_API_BASE/models" >"$result_dir/models.json"
  ./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python \
    -m external_comparison.runners.ec2_v2 \
    --repo-root "$PWD" --gdesigner-root /home/jianbaizhao/external_baselines/GDesigner \
    --data-dir /home/jianbaizhao/RPAS/data/mmlu --output-dir "$root" --method "$method" --seed "$seed" \
    --data-seed 2026 --search-per-subject 1 --select-per-subject 1 --test-per-subject 10 \
    >"$result_dir/logs/runner.log" 2>&1 &
  child=$!
  echo "$(date -Is) launched $method seed=$seed runner_wrapper=$child port=$endpoint_port"
  wait "$child"
  child=''
  py -m external_comparison.runners.write_run_metrics --result-dir "$result_dir" \
    --job-root "$result_dir" --job-id "$SLURM_JOB_ID" --stage packed_shared_endpoint_seed
  py - "$result_dir" <<'PY'
from pathlib import Path
import sys
from external_comparison.runners.aggregate_mmlu_v2 import _load_seed
result = _load_seed(Path(sys.argv[1]))
print('ARTIFACT_INTEGRITY_PASS', result['method'], result['seed'], 'truncation_gate', result['truncation_gate_passed'])
PY
  echo 'formal_result=false; seed_complete=true; shared_endpoint=true' >"$result_dir/job_status.txt"
  ( cd "$result_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS )
  echo "$(date -Is) completed $method seed=$seed"
)
run_seed gdesigner 2 "$shared_port" & workers+=("$!")
run_seed rpas_comm 1 "$port" & workers+=("$!")
run_seed rpas_comm 2 "$port" & workers+=("$!")
status=0
for pid in "${workers[@]}"; do wait "$pid" || status=1; done
workers=()
echo "$(date -Is) all bundle workers exited; status=$status"
exit "$status"
