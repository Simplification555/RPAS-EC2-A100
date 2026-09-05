#!/usr/bin/env bash
# Join an existing, externally guarded endpoint; never own or stop its service.
set -euo pipefail
[[ $# -eq 6 ]] || { echo 'usage: script <chain-seed> <port> <batch-parent> <service-pid> <activation> <owner-method>' >&2; exit 2; }
seed="$1"; port="$2"; parent="$3"; service="$4"; activation="$5"; owner_method="$6"
[[ "$seed" =~ ^[12]$ && "$port" =~ ^[0-9]+$ && "$parent" =~ ^[0-9]+$ && "$service" =~ ^[0-9]+$ ]] || exit 2
[[ "$owner_method" == full_connected && -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ && ! -e "$activation" ]] || exit 2
for pid in "$parent" "$service"; do
  tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "SLURM_JOB_ID=$SLURM_JOB_ID" >/dev/null
done
tr '\0' ' ' <"/proc/$parent/cmdline" | grep -F '/slurm_script' >/dev/null
tr '\0' ' ' <"/proc/$service/cmdline" | grep -F 'serve_transformers_qwen35_openai.py' >/dev/null
tr '\0' ' ' <"/proc/$service/cmdline" | grep -F -- "--port $port " >/dev/null
parent_start="$(awk '{print $22}' "/proc/$parent/stat")"
service_start="$(awk '{print $22}' "/proc/$service/stat")"
cd /home/jianbaizhao/RPAS-EC2-A100
root="$PWD/outputs/scir_ec2_formal"
mkdir -p "$root/_locks"
exec 9>"$root/_locks/chain_seed_${seed}.lock"
flock -n 9 || exit 2
result_dir="$root/chain/seed_$seed"
[[ ! -e "$result_dir" ]] || { echo "Refusing existing seed directory: $result_dir" >&2; exit 2; }
echo "shared_wrapper_pid=$$ seed=$seed port=$port owner_parent=$parent service=$service"
for _ in $(seq 1 600); do
  [[ -f "$activation" ]] && break
  kill -0 "$service" 2>/dev/null || exit 2
  sleep 1
done
[[ -f "$activation" ]] || { echo 'No external guard activation' >&2; exit 2; }
[[ "$(awk '{print $3, $22}' "/proc/$parent/stat")" == "T $parent_start" ]] || exit 2
[[ "$(awk '{print $22}' "/proc/$service/stat")" == "$service_start" ]] || exit 2
guard="$(<"$activation")"
[[ "$guard" =~ ^[0-9]+$ ]] || exit 2
tr '\0' '\n' <"/proc/$guard/environ" | grep -Fx "SLURM_JOB_ID=$SLURM_JOB_ID" >/dev/null
tr '\0' ' ' <"/proc/$guard/cmdline" | grep -F 'guard_packed_allocation.sh' >/dev/null
export RPAS_SCIR_ALLOCATED_GPU=1 RPAS_MMLU_MAX_TOKENS=256 RPAS_EC2_FIXED_EVAL_CONCURRENCY=4
export RPAS_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:$port/v1" RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_TOKENIZER_PATH=/home/jianbaizhao/model/Qwen/Qwen3.5-9B
export RPAS_GDESIGNER_EMBEDDING_MODEL=/home/jianbaizhao/model/sentence-transformers/all-MiniLM-L6-v2
export RPAS_MODEL_CONFIG="$PWD/experiments/phase2_mmlu_qwen35_9b.json"
export GEPA_QWEN_DISABLE_THINKING=1 GEPA_PHASE2_PROMPT_MODE=deliberate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py() { ./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python "$@"; }
mkdir -p "$result_dir/logs"
{
  echo "run_started_at_epoch=$(date +%s.%N)"
  echo "job_id=$SLURM_JOB_ID method=chain seed=$seed"
  echo "shared_endpoint=true port=$port owner_method=$owner_method owner_parent=$parent service_pid=$service guard_pid=$guard"
  echo 'timing_scope=co_resident_operational_not_isolated'
  hostname
  nvidia-smi --query-gpu=uuid,name,memory.total --format=csv
  sha256sum external_comparison/runners/ec2_v2.py scripts/scir/ec2_shared_endpoint_packed.sh
} >"$result_dir/environment.txt"
curl --max-time 5 -sf "$RPAS_EXTERNAL_API_BASE/models" >"$result_dir/models.json"
py -m external_comparison.runners.ec2_v2 \
  --repo-root "$PWD" --gdesigner-root /home/jianbaizhao/external_baselines/GDesigner \
  --data-dir /home/jianbaizhao/RPAS/data/mmlu --output-dir "$root" \
  --method chain --seed "$seed" --data-seed 2026 \
  --search-per-subject 1 --select-per-subject 1 --test-per-subject 10 \
  >"$result_dir/logs/runner.log" 2>&1
[[ -s "$result_dir/result.json" && -s "$result_dir/calls.jsonl" ]] || exit 2
py -m external_comparison.runners.write_run_metrics --result-dir "$result_dir" \
  --job-root "$result_dir" --job-id "$SLURM_JOB_ID" --stage shared_endpoint_seed
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
echo "$(date -Is) chain_seed=$seed complete; service remains owned by original batch"
